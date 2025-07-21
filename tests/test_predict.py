#!/usr/bin/env python3
"""
Tests for the prediction module.
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import FraudPredictor, create_sample_transaction

class TestFraudPredictor:
    """Test cases for FraudPredictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test models
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_sample_transaction(self):
        """Test sample transaction creation."""
        sample = create_sample_transaction()
        
        assert isinstance(sample, dict)
        assert 'V1' in sample
        assert 'Amount' in sample
        assert 'hour' in sample
        assert 'day' in sample
        assert len(sample) > 30  # Should have all required features
    
    def test_predictor_initialization_without_model(self):
        """Test predictor initialization when no model is available."""
        with pytest.raises(FileNotFoundError):
            FraudPredictor(model_dir=self.temp_dir)
    
    def test_sample_transaction_structure(self):
        """Test that sample transaction has correct structure."""
        sample = create_sample_transaction()
        
        # Check required V features
        for i in range(1, 29):
            assert f'V{i}' in sample
        
        # Check engineered features
        required_features = [
            'Amount', 'hour', 'day', 'amount_log', 'amount_sqrt',
            'v_mean', 'v_std', 'v_max', 'v_min', 'v_range',
            'amount_time_interaction', 'Amount_capped'
        ]
        
        for feature in required_features:
            assert feature in sample, f"Missing feature: {feature}"
    
    def test_sample_transaction_data_types(self):
        """Test that sample transaction has correct data types."""
        sample = create_sample_transaction()
        
        # V features should be floats
        for i in range(1, 29):
            assert isinstance(sample[f'V{i}'], float)
        
        # Amount should be float
        assert isinstance(sample['Amount'], float)
        
        # Time features should be ints
        assert isinstance(sample['hour'], int)
        assert isinstance(sample['day'], int)
        
        # Engineered features should be floats
        assert isinstance(sample['amount_log'], float)
        assert isinstance(sample['amount_sqrt'], float)
        assert isinstance(sample['v_mean'], float)
        assert isinstance(sample['v_std'], float)
        assert isinstance(sample['v_max'], float)
        assert isinstance(sample['v_min'], float)
        assert isinstance(sample['v_range'], float)
        assert isinstance(sample['amount_time_interaction'], float)
        assert isinstance(sample['Amount_capped'], float)

if __name__ == "__main__":
    pytest.main([__file__]) 