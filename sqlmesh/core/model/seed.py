from __future__ import annotations

import logging
import typing as t
import zlib
from io import StringIO
from pathlib import Path

from sqlglot import exp
from sqlglot.dialects.dialect import UNESCAPED_SEQUENCES
from sqlglot.helper import seq_get
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from sqlmesh.core.model.common import parse_bool
from sqlmesh.utils.pandas import columns_to_types_from_df
from sqlmesh.utils.pydantic import PydanticModel, field_validator

if t.TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

NaHashables = t.List[t.Union[int, str, bool, t.Literal[None]]]
NaValues = t.Union[NaHashables, t.Dict[str, NaHashables]]


class CsvSettings(PydanticModel):
    """Settings for CSV seeds."""

    delimiter: t.Optional[str] = None
    quotechar: t.Optional[str] = None
    doublequote: t.Optional[bool] = None
    escapechar: t.Optional[str] = None
    skipinitialspace: t.Optional[bool] = None
    lineterminator: t.Optional[str] = None
    encoding: t.Optional[str] = None
    na_values: t.Optional[NaValues] = None
    keep_default_na: t.Optional[bool] = None

    @field_validator("doublequote", "skipinitialspace", "keep_default_na", mode="before")
    @classmethod
    def _bool_validator(cls, v: t.Any) -> t.Optional[bool]:
        if v is None:
            return v
        return parse_bool(v)

    @field_validator(
        "delimiter", "quotechar", "escapechar", "lineterminator", "encoding", mode="before"
    )
    @classmethod
    def _str_validator(cls, v: t.Any) -> t.Optional[str]:
        if v is None or not isinstance(v, exp.Expr):
            return v

        # SQLGlot parses escape sequences like \t as \\t for dialects that don't treat \ as
        # an escape character, so we map them back to the corresponding escaped sequence
        v = v.this
        return UNESCAPED_SEQUENCES.get(v, v)

    @field_validator("na_values", mode="before")
    @classmethod
    def _na_values_validator(cls, v: t.Any) -> t.Optional[NaValues]:
        if v is None or not isinstance(v, exp.Expr):
            return v

        try:
            if isinstance(v, exp.Paren) or not isinstance(v, (exp.Tuple, exp.Array)):
                v = exp.Tuple(expressions=[v.unnest()])

            expressions = v.expressions
            if isinstance(seq_get(expressions, 0), (exp.PropertyEQ, exp.EQ)):
                return {
                    e.left.name: [
                        rhs_val.to_py()
                        for rhs_val in (
                            [e.right.unnest()]
                            if isinstance(e.right, exp.Paren)
                            else e.right.expressions
                        )
                    ]
                    for e in expressions
                }

            return [e.to_py() for e in expressions]
        except ValueError as e:
            logger.warning(f"Failed to coerce na_values '{v}', proceeding with defaults. {str(e)}")

        return None


class CsvSeedReader:
    def __init__(
        self,
        content: str,
        dialect: str,
        settings: CsvSettings,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
    ):
        self.content = content
        self.dialect = dialect
        self.settings = settings
        self._declared_columns_to_types = columns_to_types or {}
        self._df: t.Optional[pd.DataFrame] = None

    @property
    def columns_to_types(self) -> t.Dict[str, exp.DataType]:
        return columns_to_types_from_df(self._get_df())

    @property
    def column_hashes(self) -> t.Dict[str, str]:
        df = self._get_df()
        return {
            column_name: str(zlib.crc32(df[column_name].to_json().encode("utf-8")))
            for column_name in df.columns
        }

    def read(self, batch_size: t.Optional[int] = None) -> t.Generator[pd.DataFrame, None, None]:
        df = self._get_df()

        batch_size = batch_size or df.size
        batch_start = 0
        while batch_start < df.shape[0]:
            yield df.iloc[batch_start : batch_start + batch_size, :]
            batch_start += batch_size

    def _get_df(self) -> pd.DataFrame:
        import pandas as pd

        if self._df is None:
            csv_settings = self._csv_settings()
            self._df = pd.read_csv(
                StringIO(self.content),
                index_col=False,
                on_bad_lines="error",
                low_memory=False,
                **csv_settings,
                **self._read_csv_type_options(pd, csv_settings),
            )
            self._df = self._df.rename(
                columns={
                    col: normalize_identifiers(col, dialect=self.dialect).name
                    for col in self._df.columns
                },
            )

        return self._df

    def _csv_settings(self) -> t.Dict[str, t.Any]:
        return {k: v for k, v in self.settings.dict().items() if v is not None}

    def _read_csv_type_options(
        self, pd: t.Any, csv_settings: t.Dict[str, t.Any]
    ) -> t.Dict[str, t.Any]:
        if not self._declared_columns_to_types:
            return {}

        csv_columns = pd.read_csv(
            StringIO(self.content),
            index_col=False,
            on_bad_lines="error",
            nrows=0,
            **csv_settings,
        ).columns
        csv_columns_by_normalized_name = {
            normalize_identifiers(column, dialect=self.dialect).name: column
            for column in csv_columns
        }

        dtype = {}
        parse_dates = []
        for column, data_type in self._declared_columns_to_types.items():
            csv_column = csv_columns_by_normalized_name.get(
                normalize_identifiers(column, dialect=self.dialect).name
            )
            if csv_column is None:
                continue

            pandas_dtype = _pandas_dtype_for(data_type)
            if pandas_dtype:
                dtype[csv_column] = pandas_dtype
            elif _should_parse_dates(data_type):
                parse_dates.append(csv_column)

        read_csv_options = {}
        if dtype:
            read_csv_options["dtype"] = dtype
        if parse_dates:
            read_csv_options["parse_dates"] = parse_dates
        return read_csv_options


class Seed(PydanticModel):
    """Represents content of a seed.

    Presently only CSV format is supported.
    """

    content: str

    def reader(
        self,
        dialect: str = "",
        settings: t.Optional[CsvSettings] = None,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
    ) -> CsvSeedReader:
        return CsvSeedReader(self.content, dialect, settings or CsvSettings(), columns_to_types)


def _pandas_dtype_for(data_type: exp.DataType) -> t.Optional[str]:
    if data_type.this in exp.DataType.TEXT_TYPES:
        return "string"
    if data_type.is_type(*exp.DataType.INTEGER_TYPES):
        return "Int64"
    if data_type.is_type("boolean"):
        return "boolean"
    if data_type.is_type("float", "double", "decimal", "numeric", "real"):
        return "Float64"
    return None


def _should_parse_dates(data_type: exp.DataType) -> bool:
    return data_type.is_type("date", "date32", "datetime", "timestamp", "timestamptz")


def create_seed(path: str | Path) -> Seed:
    with open(Path(path), "r", encoding="utf-8") as fd:
        return Seed(content=fd.read())
