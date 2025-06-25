# agents/neuralake_agent.py

from agents.base_agent import Agent
from neuralake.core import Catalog

class NeuralakeAgent(Agent):
    """Agent to query, materialize, and manage Neuralake tables and catalogs."""
    def __init__(self,
                 catalog: Catalog,
                 db_name: str = "bci",
                 model_name: str = "gpt-4",
                 temperature: float = 0.0,
                 client=None):
        """
        Args:
            catalog: your neuralake Catalog instance
            db_name: which catalog DB to operate on (e.g. "bci")
        """
        super().__init__(catalog=catalog,
                         model_name=model_name,
                         temperature=temperature,
                         client=client)
        self.db_name = db_name

    def list_tables(self) -> list[str]:
        """What tables are currently registered?"""
        return self.catalog.db(self.db_name).list_tables()

    def get_schema(self, table_name: str):
        """Return the pyarrow.Schema for a table."""
        tbl = self.catalog.db(self.db_name).table(table_name)
        return tbl.schema

    def materialize(self, table_name: str, uri: str, mode: str = "overwrite"):
        """
        Write out a table to a URI (Parquet/Delta) so downstream consumers can use it.
        mode: 'overwrite' | 'append'
        """
        tbl = self.catalog.db(self.db_name).table(table_name)
        tbl.write(uri, mode=mode)

    def query(self, table_name: str, sql: str):
        """Run an ad-hoc SQL/Polars query against the catalog."""
        return self.catalog.db(self.db_name).sql(sql).collect()

    # def export_site(self, output_dir: str):
    #     """Generate a zero-server static site of your catalog."""
    #     from neuralake.export.web import export_and_generate_site
    #     export_and_generate_site([(self.db_name, self.catalog)], output_dir=output_dir)
    #     self.logger.info(f"Static site generated at {output_dir}")

    def generate_roapi_config(self, output_file: str = "roapi-config.yaml"):
        """Create a ROAPI config to expose your catalog via REST/GraphQL."""
        from neuralake.export import roapi
        roapi.generate_config(self.catalog, output_file=output_file)
        self.logger.info(f"ROAPI config written to {output_file}")

    def get_lineage(self, table_name: str) -> dict:
        """
        If you’ve been registering parent-child relationships in metadata,
        pull them out here to visualize or audit pipelines.
        """
        tbl = self.catalog.db(self.db_name).table(table_name)
        return {
            "depends_on": tbl.metadata.get("parents", []),
            "children": tbl.metadata.get("children", []),
        }
