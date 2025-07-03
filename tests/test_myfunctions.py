# File: tests/test_myfunctions.py
import pytest
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from myfunctions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType

# Create Spark session for testing
spark = SparkSession.builder \
    .appName('unit-tests') \
    .getOrCreate()

# Test data schemas
customer_schema = StructType([
    StructField("customer_id", IntegerType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("status", StringType(), True)
])

transaction_schema = StructType([
    StructField("customer_id", IntegerType(), True),
    StructField("transaction_id", StringType(), True),
    StructField("transaction_amount", FloatType(), True)
])

# Test data
customer_data = [
    (1, "john", "doe", "john.doe@email.com ", "(555) 123-4567", "active"),
    (2, "jane", "smith", " jane.smith@email.com", "555-987-6543", None),
    (3, "bob", "johnson", "bob@email.com", "5551234567", "inactive")
]

transaction_data = [
    (1, "T001", 100.0),
    (1, "T002", 200.0),
    (2, "T003", 150.0),
    (2, "T004", 250.0),
    (2, "T005", 100.0)
]

# Utility function to create test DataFrames
def create_customer_df():
    return spark.createDataFrame(customer_data, customer_schema)

def create_transaction_df():
    return spark.createDataFrame(transaction_data, transaction_schema)

# Unit Tests
def test_columnExists():
    """Test column existence check"""
    df = create_customer_df()
    assert columnExists(df, "customer_id") is True
    assert columnExists(df, "nonexistent_column") is False
    print("✅ test_columnExists passed")

def test_numRowsInColumnForValue():
    """Test row counting for specific values"""
    df = create_customer_df()
    assert numRowsInColumnForValue(df, "status", "active") == 1
    assert numRowsInColumnForValue(df, "status", "inactive") == 1
    assert numRowsInColumnForValue(df, "status", "nonexistent") == 0
    print("✅ test_numRowsInColumnForValue passed")

def test_clean_customer_data():
    """Test customer data cleaning function"""
    df = create_customer_df()
    result_df = clean_customer_data(df)
    
    # Collect results for assertion
    results = result_df.collect()
    
    # Check first row transformations
    first_row = results[0]
    assert first_row.first_name == "JOHN"
    assert first_row.last_name == "DOE"
    assert first_row.email == "john.doe@email.com"  # Space removed
    assert first_row.phone == "5551234567"  # Special chars removed
    
    # Check null handling
    second_row = results[1]
    assert second_row.status == "ACTIVE"  # NULL replaced with ACTIVE
    
    print("✅ test_clean_customer_data passed")

def test_calculate_customer_metrics():
    """Test customer metrics calculation"""
    df = create_transaction_df()
    result_df = calculate_customer_metrics(df)
    
    results = result_df.collect()
    
    # Convert to dictionary for easier testing
    results_dict = {row.customer_id: row for row in results}
    
    # Customer 1: 2 transactions, total=300, avg=150
    customer_1 = results_dict[1]
    assert customer_1.total_spent == 300.0
    assert customer_1.avg_transaction == 150.0
    assert customer_1.transaction_count == 2
    
    # Customer 2: 3 transactions, total=500, avg=166.67
    customer_2 = results_dict[2]
    assert customer_2.total_spent == 500.0
    assert abs(customer_2.avg_transaction - 166.67) < 0.01
    assert customer_2.transaction_count == 3
    
    print("✅ test_calculate_customer_metrics passed")

def test_validate_data_quality():
    """Test data quality validation"""
    df = create_customer_df()
    
    # Test with all required columns present
    required_columns = ["customer_id", "first_name", "last_name", "email"]
    result = validate_data_quality(df, required_columns)
    
    assert result["is_valid"] is True
    assert len(result["issues"]) == 0
    assert result["total_rows"] == 3
    
    # Test with missing columns
    required_columns_missing = ["customer_id", "nonexistent_column"]
    result_missing = validate_data_quality(df, required_columns_missing)
    
    assert result_missing["is_valid"] is False
    assert len(result_missing["issues"]) > 0
    
    print("✅ test_validate_data_quality passed")

# Test runner functions
def run_all_tests():
    """Run all tests manually"""
    try:
        test_columnExists()
        test_numRowsInColumnForValue()
        test_clean_customer_data()
        test_calculate_customer_metrics()
        test_validate_data_quality()
        print("\n🎉 All tests passed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    run_all_tests()