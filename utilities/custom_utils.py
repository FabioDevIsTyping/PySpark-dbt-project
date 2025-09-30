from typing import List, Optional, Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

class Transformations:
    """
    Helpers for record deduplication and upsert into Silver tables.
    """

    def deduplicate(
        self,
        df: DataFrame,
        dedup_cols: List[str],
        cdc: str,
        tie_breaker: Optional[str] = None
    ) -> DataFrame:
        """
        Return one row per group defined by dedup_cols, keeping the latest record.
        """
        order_cols = [F.col(cdc).desc()]
        if tie_breaker:
            order_cols.append(F.col(tie_breaker).desc())
        w = Window.partitionBy(*[F.col(c) for c in dedup_cols]).orderBy(*order_cols)
        return (
            df.withColumn("_rn", F.row_number().over(w))
              .filter(F.col("_rn") == 1)
              .drop("_rn")
        )

    def process_timestamp(self, df: DataFrame) -> DataFrame:
        """
        Add a processed_timestamp column to the DataFrame.
        """
        return df.withColumn("processed_timestamp", current_timestamp())

    def upsert(
        self,
        df: DataFrame,
        key_cols: List[str],
        target_table: str,
        cdc: str
    ) -> None:
        """
        Upsert df into target_table using key_cols as the match keys.
        Only update matched rows if the source is as-new or newer on the CDC column.
        Insert all non-matching rows.
        """
        spark = df.sparkSession

        if not spark.catalog.tableExists(target_table):
            df.write.mode("overwrite").format("delta").saveAsTable(target_table)
            return

        tgt = DeltaTable.forName(spark, target_table)
        on_expr = " AND ".join([f"t.{k} = s.{k}" for k in key_cols])

        (
            tgt.alias("t")
               .merge(df.alias("s"), on_expr)
               .whenMatchedUpdateAll(condition=f"s.{cdc} >= t.{cdc}")
               .whenNotMatchedInsertAll()
               .execute()
        )

    def duplicates_report(self, df: DataFrame, keys: List[str]) -> DataFrame:
        """
        Return a DataFrame with the number of duplicates per group defined by keys.
        """
        return df.groupBy(*[F.col(k) for k in keys]).count().filter(F.col("count") > 1)
    
    def deduplicate_by_recency(self, df: DataFrame, keys: List[str], cdc: str, tie_breaker: Optional[str] = None) -> DataFrame:
        """
        Keep the latest row per key ordered by cdc desc and optional tie_breaker desc.
        """
        order_cols = [F.col(cdc).desc()]
        if tie_breaker:
            order_cols.append(F.col(tie_breaker).desc())
        w = Window.partitionBy(*[F.col(k) for k in keys]).orderBy(*order_cols)
        return df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
    
    def nulls_report(self, df: DataFrame, cols: List[str]) -> DataFrame:
        """
        Return counts and ratios of nulls per requested columns.
        """
        total = df.count()
        aggs = []
        for c in cols:
            aggs.append(F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls"))
        res = df.agg(*aggs)
        ratio_exprs = [(F.col(f"{c}_nulls") / F.lit(total)).alias(f"{c}_null_ratio") for c in cols]
        return res.select(*res.columns, *ratio_exprs)

    def require_non_null(self, df: DataFrame, required_cols: List[str]) -> Dict[str, DataFrame]:
        """
        Split dataframe into valid and rejected based on non-null required columns.
        """
        cond = None
        for c in required_cols:
            expr = F.col(c).isNotNull()
            cond = expr if cond is None else (cond & expr)
        valid = df.filter(cond)
        rejected = df.filter(~cond)
        return {"valid": valid, "rejected": rejected}

    def filter_by_ranges(self, df: DataFrame, ranges: Dict[str, Tuple[float, float]], inclusive: bool = True) -> Dict[str, DataFrame]:
        """
        Split dataframe into valid and rejected by enforcing numeric ranges on one or more columns.
        """
        cond = None
        for c, bounds in ranges.items():
            lo, hi = bounds
            cold = F.col(c).cast("double")
            expr = cold.between(F.lit(lo), F.lit(hi)) if inclusive else ((cold > F.lit(lo)) & (cold < F.lit(hi)))
            cond = expr if cond is None else (cond & expr)
        valid = df.filter(cond) if cond is not None else df
        rejected = df.filter(~cond) if cond is not None else df.limit(0)
        return {"valid": valid, "rejected": rejected}
    
    def sanitize_string(self, df:DataFrame, cols: List[str], pattern: str = r"[ \-]", replacement: str = "") -> DataFrame:
        """
        Sanitize strings by replacing pattern with replacement.
        """
        for c in cols:
            df = df.withColumn(c, F.regexp_replace(F.col(c), pattern, replacement))
        return df