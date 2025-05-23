import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_pyarrow_mmap():
    # Create a sample table
    data = {"column1": [i for i in range(10000)], "column2": [i for i in range(10000)]}
    table = pa.table(data)

    # Write the table to a Parquet dataset on disk
    dataset_path = "example_dataset.parquet"
    pq.write_table(table, dataset_path)

    # Read the table back using memory mapping
    t0 = time.perf_counter()
    memory_mapped_table = pq.read_table(dataset_path, memory_map=True)

    # Verify the contents
    print(memory_mapped_table)
    t1 = time.perf_counter()
    print(f"Reading the table took {t1 - t0:.2f} seconds.")


def write_multiple_tables():
    # Define the schema
    fields = [
        pa.field("id", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("age", pa.int32()),
        pa.field(
            "address",
            pa.struct(
                [pa.field("street", pa.string()), pa.field("city", pa.string()), pa.field("zipcode", pa.int32())]
            ),
        ),
        pa.field("contacts", pa.list_(pa.struct([pa.field("type", pa.string()), pa.field("value", pa.string())]))),
    ]

    schema = pa.schema(fields)
    print(schema)

    # Initialize ParquetWriter
    file_path = "large_dataset.parquet"
    writer = pq.ParquetWriter(file_path, schema)

    # Example function to generate a batch of data
    def generate_batch(start, end):
        return pd.DataFrame(
            {
                "id": range(start, end),
                "name": ["Name" + str(i) for i in range(start, end)],
                "age": [20 + (i % 30) for i in range(start, end)],
                "address": [
                    {"street": f"{i} Main St", "city": "City", "zipcode": 10000 + i} for i in range(start, end)
                ],
                "contacts": [
                    [{"type": "email", "value": f"name{i}@example.com"}, {"type": "phone", "value": f"555-{i:04d}"}]
                    for i in range(start, end)
                ],
            }
        )

    # Write data in batches
    batch_size = 10000  # Adjust batch size as needed
    total_rows = 1000000  # Adjust total number of rows as needed

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = generate_batch(start, end)
        table = pa.Table.from_pandas(batch, schema=schema)
        writer.write_table(table)
        print(f"Wrote {end} rows")

    # Close the writer
    writer.close()

    # Read the table back using memory mapping
    t0 = time.perf_counter()
    memory_mapped_table = pq.read_table(file_path, memory_map=True)
    # Count rows
    print(memory_mapped_table.num_rows)
    t1 = time.perf_counter()
    print(f"Reading the table took {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    # test_pyarrow_mmap()
    write_multiple_tables()
