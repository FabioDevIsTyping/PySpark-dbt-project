from typing import List, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

class Transformations:
    """
    Helpers for record deduplication in Silver tables.
    """

    def deduplicate(self, df: DataFrame, dedup_cols: List[str], cdc: str, tie_breaker: Optional[str] = None) -> DataFrame:
        """
        Return one row per group defined by dedup_cols, keeping the latest record.
        
        Parameters
        df: Input DataFrame.
        dedup_cols: Columns that define a unique entity, for example ["customer_id"].
        cdc: Change detection column used for recency ordering, typically an updated_at or event timestamp.
        tie_breaker: Optional secondary column to break ties when cdc values are equal, for example ingestion_timestamp.
        
        Behavior
        Within each group defined by dedup_cols, rows are ordered by cdc descending and then by tie_breaker descending if provided. 
        The first row per group is kept and the others are discarded.
        """
        order_cols = [F.col(cdc).desc()]
        if tie_breaker:
            order_cols.append(F.col(tie_breaker).desc())
        w = Window.partitionBy(*[F.col(c) for c in dedup_cols]).orderBy(*order_cols)
        return df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
    
    def process_timestamp(self,df):
        """
        Add a	processed_timestamp column to the dataframe
        """
        df = df.withColumn("processed_timestamp", current_timestamp())
        return df
    

        def upsert(self, df: DataFrame, key_cols: List[str], target_table: str, cdc: str) -> None:
        """
        Upsert df into target_table using key_cols as the match keys.
        Only update matched rows if the source is as-new or newer on the CDC column.
        Insert all non-matching rows.
        """
        spark = df.sparkSession

        # Create table on first run
        if not spark._jsparkSession.catalog().tableExists(target_table):
            df.write.mode("overwrite").format("delta").saveAsTable(target_table)
            return

        tgt = DeltaTable.forName(spark, target_table)
        on_expr = " AND ".join([f"t.{k} = s.{k}" for k in key_cols])

        (tgt.alias("t")
            .merge(df.alias("s"), on_expr)
            .whenMatchedUpdateAll(condition=f"s.{cdc} >= t.{cdc}")   # update only if source is newer
            .whenNotMatchedInsertAll()                               # insert new keys
            .execute())