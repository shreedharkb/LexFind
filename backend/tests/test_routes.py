import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import create_app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app with mocked search service."""
    with patch('app.services.qdrant_search_service.get_search_service') as mock_get_searcher:
        mock_searcher = MagicMock()
        mock_get_searcher.return_value = mock_searcher
        
        app = create_app()
        yield TestClient(app)

def test_health_endpoint(client):
    """Test the /api/health endpoint."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "message" in data
    assert "total_cases" in data
    
    assert data["status"] == "healthy"
    assert "Legal case search service is running" in data["message"]
    assert isinstance(data["total_cases"], int)
    assert data["total_cases"] >= 0