"""
Step 4: Import Data from CSV
This script demonstrates how to import data into the knowledge graph from CSV files
"""

from neo4j import GraphDatabase
import pandas as pd
import os


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]


def preview_csv_data(csv_path):
    """Preview the CSV data before importing"""
    print(f"\nPreviewing CSV data from: {csv_path}")
    print("-" * 60)

    df = pd.read_csv(csv_path)
    print(df.to_string(index=False))
    print(f"\nTotal rows: {len(df)}")
    print("-" * 60)

    return df


def import_from_dataframe(conn, df):
    """Import data from pandas DataFrame"""
    print("\nImporting data from DataFrame...")

    imported_count = 0

    for _, row in df.iterrows():
        # Create or merge person
        person_query = """
        MERGE (p:Person {name: $person_name})
        RETURN p.name AS name
        """
        conn.query(person_query, {"person_name": row['person_name']})

        # Create or merge movie
        movie_query = """
        MERGE (m:Movie {title: $movie_title})
        ON CREATE SET m.release_year = $release_year, m.genre = $genre
        RETURN m.title AS title
        """
        conn.query(movie_query, {
            "movie_title": row['movie_title'],
            "release_year": int(row['release_year']),
            "genre": row['genre']
        })

        # Create relationship with role
        relationship_query = """
        MATCH (p:Person {name: $person_name})
        MATCH (m:Movie {title: $movie_title})
        MERGE (p)-[r:ACTED_IN]->(m)
        SET r.role = $role
        RETURN p.name AS actor, m.title AS movie, r.role AS role
        """
        result = conn.query(relationship_query, {
            "person_name": row['person_name'],
            "movie_title": row['movie_title'],
            "role": row['role']
        })

        if result:
            print(
                f"  ✓ {result[0]['actor']} -[ACTED_IN: {result[0]['role']}]-> {result[0]['movie']}")
            imported_count += 1

    print(f"\nSuccessfully imported {imported_count} relationships")


def show_updated_statistics(conn):
    """Display updated graph statistics"""
    print("\nUpdated Graph Statistics:")

    result = conn.query("MATCH (p:Person) RETURN count(p) as count")
    print(f"  Total Person nodes: {result[0]['count']}")

    result = conn.query("MATCH (m:Movie) RETURN count(m) as count")
    print(f"  Total Movie nodes: {result[0]['count']}")

    result = conn.query("MATCH ()-[r:ACTED_IN]->() RETURN count(r) as count")
    print(f"  Total ACTED_IN relationships: {result[0]['count']}")

    result = conn.query("MATCH ()-[r:DIRECTED]->() RETURN count(r) as count")
    print(f"  Total DIRECTED relationships: {result[0]['count']}")


def show_recent_movies(conn):
    """Show recently added movies"""
    print("\nRecently Added Movies (2020+):")

    query = """
    MATCH (m:Movie)
    WHERE m.release_year >= 2020
    RETURN m.title AS title, m.release_year AS year, m.genre AS genre
    ORDER BY m.release_year DESC
    """

    results = conn.query(query)
    for record in results:
        print(f"  • {record['title']} ({record['year']}) - {record['genre']}")


def main():
    print("=" * 60)
    print("STEP 4: Importing Data from CSV")
    print("=" * 60)

    # Connection parameters
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "password123"

    # CSV file path
    csv_path = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'movie_data.csv')

    # Connect to Neo4j
    conn = Neo4jConnection(URI, USER, PASSWORD)

    try:
        # Preview CSV data
        df = preview_csv_data(csv_path)

        # Import data
        import_from_dataframe(conn, df)

        # Show updated statistics
        show_updated_statistics(conn)

        # Show recent movies
        show_recent_movies(conn)

        print("\n" + "=" * 60)
        print("Step 4 completed successfully!")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
