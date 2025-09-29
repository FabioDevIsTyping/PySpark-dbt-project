from typing import List, Optional
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
