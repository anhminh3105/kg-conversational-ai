"""
Data loader for PrimeKG (Precision Medicine Knowledge Graph).

Reads kg.csv from PrimeKG and converts rows into Triplet objects compatible
with the rest of the RAG pipeline. Supports filtering by node types and
relation types, and row limits for working with subsets of the ~8.1M-row
dataset.

PrimeKG CSV columns:
    relation, display_relation, x_index, x_id, x_type, x_name, x_source,
    y_index, y_id, y_type, y_name, y_source

Mapping to Triplet:
    x_name  -> subject
    display_relation -> predicate
    y_name  -> object
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from .triplet_loader import Triplet

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "display_relation", "x_name", "y_name",
    "x_type", "y_type", "x_id", "y_id", "x_source", "y_source",
}


class PrimeKGLoader:
    """
    Loads and parses triplets from a PrimeKG kg.csv file.

    Usage:
        loader = PrimeKGLoader("data/kg.csv", node_types=["drug", "disease"])
        triplets = loader.load().parse()
    """

    def __init__(
        self,
        path: str,
        node_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ):
        """
        Args:
            path: Path to kg.csv
            node_types: Keep only rows where x_type OR y_type is in this list
            relation_types: Keep only rows whose display_relation is in this list
            max_rows: Maximum number of CSV rows to read (applied before filtering)
        """
        self.path = Path(path)
        self.node_types: Optional[Set[str]] = (
            {t.lower().strip() for t in node_types} if node_types else None
        )
        self.relation_types: Optional[Set[str]] = (
            {r.lower().strip() for r in relation_types} if relation_types else None
        )
        self.max_rows = max_rows

        self._df: Optional[pd.DataFrame] = None
        self.triplets: List[Triplet] = []

    # ------------------------------------------------------------------
    # Public API (mirrors TripletLoader interface)
    # ------------------------------------------------------------------

    def load(self) -> "PrimeKGLoader":
        """Read the CSV into memory. Returns self for chaining."""
        csv_path = self._resolve_path()
        logger.info(f"Loading PrimeKG from {csv_path}")

        self._df = pd.read_csv(csv_path, nrows=self.max_rows)

        missing = REQUIRED_COLUMNS - set(self._df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required PrimeKG columns: {missing}. "
                f"Found columns: {list(self._df.columns)}"
            )

        logger.info(
            f"Loaded {len(self._df)} rows from PrimeKG "
            f"({self._df['x_type'].nunique()} node types, "
            f"{self._df['display_relation'].nunique()} relation types)"
        )
        return self

    def parse(
        self,
        deduplicate: bool = True,
        normalize: bool = True,
    ) -> List[Triplet]:
        """
        Convert loaded rows to Triplet objects.

        Args:
            deduplicate: Remove duplicate triplets
            normalize: Replace underscores with spaces in entity names
        """
        if self._df is None:
            raise ValueError("Must call load() before parse()")

        df = self._df

        # --- apply filters ---
        if self.node_types is not None:
            mask = (
                df["x_type"].str.lower().isin(self.node_types)
                | df["y_type"].str.lower().isin(self.node_types)
            )
            before = len(df)
            df = df[mask]
            logger.info(
                f"Filtered by node types {self.node_types}: "
                f"{before} -> {len(df)} rows"
            )

        if self.relation_types is not None:
            mask = df["display_relation"].str.lower().isin(self.relation_types)
            before = len(df)
            df = df[mask]
            logger.info(
                f"Filtered by relation types {self.relation_types}: "
                f"{before} -> {len(df)} rows"
            )

        # --- convert to Triplet objects ---
        self.triplets = []
        for _, row in df.iterrows():
            source_text = (
                f"x_type={row['x_type']}; y_type={row['y_type']}; "
                f"x_id={row['x_id']}; y_id={row['y_id']}; "
                f"x_source={row.get('x_source', '')}; y_source={row.get('y_source', '')}"
            )
            self.triplets.append(
                Triplet(
                    subject=str(row["x_name"]),
                    predicate=str(row["display_relation"]),
                    obj=str(row["y_name"]),
                    source_text=source_text,
                )
            )

        raw_count = len(self.triplets)
        logger.info(f"Parsed {raw_count} triplets from PrimeKG")

        if normalize:
            self.triplets = [t.normalized() for t in self.triplets]
            logger.info("Normalized entity names")

        if deduplicate:
            original = len(self.triplets)
            self.triplets = list(dict.fromkeys(self.triplets))
            removed = original - len(self.triplets)
            if removed > 0:
                logger.info(
                    f"Removed {removed} duplicates ({len(self.triplets)} unique)"
                )

        return self.triplets

    # ------------------------------------------------------------------
    # Exploration helpers
    # ------------------------------------------------------------------

    def get_unique_node_types(self) -> List[str]:
        """Return sorted list of unique node types present in the loaded data."""
        if self._df is None:
            raise ValueError("Must call load() first")
        types = set(self._df["x_type"].unique()) | set(self._df["y_type"].unique())
        return sorted(types)

    def get_unique_relations(self) -> List[str]:
        """Return sorted list of unique display_relation values."""
        if self._df is None:
            raise ValueError("Must call load() first")
        return sorted(self._df["display_relation"].unique())

    def get_triplets_by_subject(self, subject: str) -> List[Triplet]:
        """Get all triplets for a given subject entity."""
        return [t for t in self.triplets if t.subject.lower() == subject.lower()]

    def get_triplets_by_relation(self, relation: str) -> List[Triplet]:
        """Get all triplets for a given relation/predicate."""
        return [t for t in self.triplets if t.predicate.lower() == relation.lower()]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        path: str,
        node_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> "PrimeKGLoader":
        """Load and parse in one call."""
        loader = cls(
            path,
            node_types=node_types,
            relation_types=relation_types,
            max_rows=max_rows,
        )
        loader.load()
        loader.parse()
        return loader

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_path(self) -> Path:
        """Resolve the path to the actual kg.csv file."""
        if self.path.is_file():
            return self.path

        if self.path.is_dir():
            kg_csv = self.path / "kg.csv"
            if kg_csv.exists():
                return kg_csv

        raise FileNotFoundError(
            f"Could not find PrimeKG CSV at {self.path}. "
            "Provide a path to kg.csv or a directory containing it. "
            "Use scripts/download_primekg.py to download the dataset."
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python primekg_loader.py <path_to_kg.csv> [--max_rows N]")
        sys.exit(1)

    path = sys.argv[1]
    max_rows = None
    if "--max_rows" in sys.argv:
        max_rows = int(sys.argv[sys.argv.index("--max_rows") + 1])

    loader = PrimeKGLoader(path, max_rows=max_rows)
    triplets = loader.load().parse()

    print(f"\nLoaded {len(triplets)} triplets")
    print(f"Node types: {loader.get_unique_node_types()}")
    print(f"Unique relations: {len(loader.get_unique_relations())}")

    print("\nSample triplets:")
    for t in triplets[:10]:
        print(f"  {t}")
