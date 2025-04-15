"""
Tests for the conda_env_executor.hypha_service module.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import functions to test
from conda_env_executor.hypha_service import (
    execute_in_conda_env,
    submit_job,
    get_job_status,
    get_job_result,
    wait_for_result,
    list_jobs,
    cancel_job,
    job_queue  # Access the global job queue instance
)
from conda_env_executor.executor import ExecutionResult, TimingInfo
from conda_env_executor.job_queue import JobStatus, JobInfo, JobResult

# Fixtures
@pytest.fixture
def mock_context():
    """Provides a mock context dictionary."""
    return {"user": {"id": "test_user"}}

@pytest.fixture(autouse=True)
def mock_job_queue():
    """Mocks the global job_queue instance for each test."""
    with patch('conda_env_executor.hypha_service.job_queue', new_callable=MagicMock) as mock_queue:
        # Configure async methods
        mock_queue.submit_job = AsyncMock()
        mock_queue.get_job_status = AsyncMock()
        mock_queue.get_job_result = AsyncMock()
        mock_queue.wait_for_result = AsyncMock()
        mock_queue.list_jobs = AsyncMock()
        mock_queue.cancel_job = AsyncMock()
        yield mock_queue

@pytest.fixture(autouse=True)
def mock_conda_env_executor():
    """Mocks the CondaEnvExecutor class."""
    with patch('conda_env_executor.hypha_service.CondaEnvExecutor', new_callable=MagicMock) as mock_executor_cls:
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = MagicMock()
        mock_executor_instance.cleanup = MagicMock()
        mock_executor_instance.env_path = "/fake/env/path"

        mock_executor_cls.create_temp_env = MagicMock(return_value=mock_executor_instance)
        yield mock_executor_cls

@pytest.fixture
def sample_code():
    """Provides sample Python code."""
    return "def execute(input_data):\\n    return input_data * 2"

@pytest.fixture
def sample_input_data():
    """Provides sample input data."""
    return 5

@pytest.fixture
def sample_dependencies():
    """Provides sample dependencies."""
    return ["python=3.9", "numpy"]

@pytest.fixture
def sample_channels():
    """Provides sample channels."""
    return ["conda-forge"]

@pytest.fixture
def sample_timing_info():
    """Provides sample TimingInfo."""
    return TimingInfo(env_setup_time=10.5, execution_time=1.5, total_time=12.0)

@pytest.fixture
def sample_job_id():
    """Provides a sample job ID."""
    return "test-job-123"

# Test Cases

# --- Tests for execute_in_conda_env ---
@pytest.mark.asyncio
async def test_execute_in_conda_env_success(
    mock_context, sample_code, sample_input_data, 
    sample_dependencies, sample_channels, sample_timing_info,
    mock_conda_env_executor
):
    """Test successful execution of code."""
    # Configure mocks
    mock_executor_instance = mock_conda_env_executor.create_temp_env.return_value
    mock_executor_instance.execute.return_value = ExecutionResult(
        success=True, 
        result=sample_input_data * 2, 
        timing=sample_timing_info
    )

    # Call the function
    result = await execute_in_conda_env(
        code=sample_code,
        input_data=sample_input_data,
        dependencies=sample_dependencies,
        channels=sample_channels,
        context=mock_context,
    )

    # Assertions
    mock_conda_env_executor.create_temp_env.assert_called_once_with(sample_dependencies, sample_channels)
    mock_executor_instance.execute.assert_called_once_with(sample_code, sample_input_data)
    mock_executor_instance.cleanup.assert_called_once()

    assert result["success"] is True
    assert result["result"] == sample_input_data * 2
    assert result["error"] is None
    assert "timing" in result
    assert result["timing"]["env_setup_time"] == sample_timing_info.env_setup_time
    assert result["timing"]["execution_time"] == sample_timing_info.execution_time
    assert result["timing"]["total_time"] == sample_timing_info.total_time

@pytest.mark.asyncio
async def test_execute_in_conda_env_defaults(
    mock_context, sample_code, sample_input_data, mock_conda_env_executor, sample_timing_info
):
    """Test execution with default dependencies and channels."""
    mock_executor_instance = mock_conda_env_executor.create_temp_env.return_value
    mock_executor_instance.execute.return_value = ExecutionResult(
        success=True,
        result=10,
        timing=sample_timing_info
    )

    result = await execute_in_conda_env(
        code=sample_code,
        input_data=sample_input_data,
        context=mock_context,
    )

    mock_conda_env_executor.create_temp_env.assert_called_once_with(["python"], ["conda-forge"])
    assert result["success"] is True
    assert result["result"] == 10
    assert result["timing"]["env_setup_time"] == sample_timing_info.env_setup_time

@pytest.mark.asyncio
async def test_execute_in_conda_env_no_code(mock_context):
    """Test execution with no code provided."""
    result = await execute_in_conda_env(
        code="",
        input_data=None,
        context=mock_context,
    )
    assert result["success"] is False
    assert result["error"] == "No code provided."

@pytest.mark.asyncio
async def test_execute_in_conda_env_execution_error(
    mock_context, sample_code, sample_input_data, mock_conda_env_executor, sample_timing_info
):
    """Test execution where the user code fails."""
    mock_executor_instance = mock_conda_env_executor.create_temp_env.return_value
    mock_executor_instance.execute.return_value = ExecutionResult(
        success=False,
        error="User code failed",
        stderr="Traceback...",
        timing=sample_timing_info
    )

    result = await execute_in_conda_env(
        code=sample_code,
        input_data=sample_input_data,
        context=mock_context,
    )

    assert result["success"] is False
    assert result["error"] == "User code failed"
    assert result["stderr"] == "Traceback..."
    assert result["timing"]["env_setup_time"] == sample_timing_info.env_setup_time

@pytest.mark.asyncio
async def test_execute_in_conda_env_setup_error(
    mock_context, sample_code, sample_input_data, mock_conda_env_executor
):
    """Test execution where environment setup fails."""
    mock_conda_env_executor.create_temp_env.side_effect = Exception("Failed to create env")
    mock_executor_instance = mock_conda_env_executor.create_temp_env.return_value

    result = await execute_in_conda_env(
        code=sample_code,
        input_data=sample_input_data,
        context=mock_context,
    )

    assert result["success"] is False
    assert "Service execution failed: Failed to create env" in result["error"]
    mock_executor_instance.execute.assert_not_called()
    # Cleanup might be called depending on where the exception occurred
    # In this mock setup, create_temp_env raises, so cleanup won't be called on the instance
    # mock_executor_instance.cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_execute_in_conda_env_cleanup_error(
    mock_context, sample_code, sample_input_data, mock_conda_env_executor, sample_timing_info
):
    """Test execution where cleanup fails (should still return results)."""
    mock_executor_instance = mock_conda_env_executor.create_temp_env.return_value
    mock_executor_instance.execute.return_value = ExecutionResult(
        success=True,
        result=10,
        timing=sample_timing_info
    )
    mock_executor_instance.cleanup.side_effect = Exception("Cleanup failed")

    result = await execute_in_conda_env(
        code=sample_code,
        input_data=sample_input_data,
        context=mock_context,
    )

    assert result["success"] is True
    assert result["result"] == 10
    assert result["timing"]["env_setup_time"] == sample_timing_info.env_setup_time
    mock_executor_instance.cleanup.assert_called_once()

# --- Tests for submit_job ---
@pytest.mark.asyncio
async def test_submit_job_success(
    mock_context, mock_job_queue, sample_code, sample_input_data,
    sample_dependencies, sample_channels, sample_job_id
):
    """Test successful job submission."""
    # Configure mock job queue
    mock_job_queue.submit_job.return_value = sample_job_id
    mock_job_queue.get_job_status.return_value = {
        "job_id": sample_job_id,
        "status": "pending",
        "created_at": 12345.678
    }

    # Call the function
    result = await submit_job(
        code=sample_code,
        input_data=sample_input_data,
        dependencies=sample_dependencies,
        channels=sample_channels,
        context=mock_context,
    )

    # Assertions
    mock_job_queue.submit_job.assert_called_once_with(
        code=sample_code,
        input_data=sample_input_data,
        dependencies=sample_dependencies,
        channels=sample_channels,
        user_id=mock_context["user"]["id"]
    )
    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert "status" in result
    assert result["status"]["job_id"] == sample_job_id

@pytest.mark.asyncio
async def test_submit_job_defaults(
    mock_context, mock_job_queue, sample_code, sample_input_data, sample_job_id
):
    """Test job submission with default dependencies and channels."""
    mock_job_queue.submit_job.return_value = sample_job_id
    mock_job_queue.get_job_status.return_value = {
        "job_id": sample_job_id,
        "status": "pending",
        "created_at": 12345.678
    }

    result = await submit_job(
        code=sample_code,
        input_data=sample_input_data,
        context=mock_context,
    )

    mock_job_queue.submit_job.assert_called_once_with(
        code=sample_code,
        input_data=sample_input_data,
        dependencies=["python"],
        channels=["conda-forge"],
        user_id=mock_context["user"]["id"]
    )
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert result["status"]["status"] == "pending"

@pytest.mark.asyncio
async def test_submit_job_no_code(mock_context, mock_job_queue):
    """Test job submission with no code."""
    result = await submit_job(
        code="",
        context=mock_context,
    )
    assert result["success"] is False
    assert result["error"] == "No code provided."
    mock_job_queue.submit_job.assert_not_called()

@pytest.mark.asyncio
async def test_submit_job_queue_error(mock_context, mock_job_queue, sample_code):
    """Test job submission when the job queue fails."""
    mock_job_queue.submit_job.side_effect = Exception("Queue submit failed")

    result = await submit_job(
        code=sample_code,
        context=mock_context,
    )

    assert result["success"] is False
    assert "Job submission failed: Queue submit failed" in result["error"]
    mock_job_queue.get_job_status.assert_not_called()

# --- Tests for get_job_status ---
@pytest.mark.asyncio
async def test_get_job_status_success(mock_context, mock_job_queue, sample_job_id):
    """Test getting status for an existing job."""
    status_response = {
        "job_id": sample_job_id,
        "status": "running",
        "created_at": 12345.678,
        "user_id": mock_context["user"]["id"]
    }
    mock_job_queue.get_job_status.return_value = status_response

    result = await get_job_status(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert result["status"] == status_response

@pytest.mark.asyncio
async def test_get_job_status_not_found(mock_context, mock_job_queue, sample_job_id):
    """Test getting status for a non-existent job."""
    mock_job_queue.get_job_status.return_value = None

    result = await get_job_status(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    assert result["success"] is False
    assert result["error"] == f"Job {sample_job_id} not found"

@pytest.mark.asyncio
async def test_get_job_status_queue_error(mock_context, mock_job_queue, sample_job_id):
    """Test getting status when the job queue raises an error."""
    mock_job_queue.get_job_status.side_effect = Exception("Queue status failed")

    result = await get_job_status(job_id=sample_job_id, context=mock_context)

    assert result["success"] is False
    assert "Failed to get job status: Queue status failed" in result["error"]

# --- Tests for get_job_result ---
@pytest.mark.asyncio
async def test_get_job_result_success_completed(
    mock_context, mock_job_queue, sample_job_id
):
    """Test getting result for a completed job."""
    mock_job_queue.get_job_status.return_value = {
        "job_id": sample_job_id,
        "status": "completed",
        "created_at": 12345.678,
        "user_id": mock_context["user"]["id"]
    }
    mock_job_queue.get_job_result.return_value = {
        "job_id": sample_job_id,
        "success": True,
        "result": "final_output",
        "stdout": "some output"
    }

    result = await get_job_result(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    mock_job_queue.get_job_result.assert_called_once_with(sample_job_id)
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert result["result"]["result"] == "final_output"

@pytest.mark.asyncio
async def test_get_job_result_success_failed(
    mock_context, mock_job_queue, sample_job_id
):
    """Test getting result for a job that failed."""
    mock_job_queue.get_job_status.return_value = {
        "job_id": sample_job_id,
        "status": "failed",
        "created_at": 12345.678,
        "user_id": mock_context["user"]["id"],
        "error": "Job failed during execution"
    }
    mock_job_queue.get_job_result.return_value = {
        "job_id": sample_job_id,
        "success": False,
        "error": "Job failed during execution",
        "stderr": "traceback"
    }

    result = await get_job_result(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    mock_job_queue.get_job_result.assert_called_once_with(sample_job_id)
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert result["result"]["success"] is False
    assert result["result"]["error"] == "Job failed during execution"

@pytest.mark.asyncio
async def test_get_job_result_not_completed(mock_context, mock_job_queue, sample_job_id):
    """Test getting result for a job that is still running."""
    status_response = {
        "job_id": sample_job_id,
        "status": "running",
        "created_at": 12345.678,
        "user_id": mock_context["user"]["id"]
    }
    mock_job_queue.get_job_status.return_value = status_response

    result = await get_job_result(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    mock_job_queue.get_job_result.assert_not_called()
    assert result["success"] is False
    assert f"Job {sample_job_id} is not completed yet" in result["error"]
    assert result["status"] == status_response

@pytest.mark.asyncio
async def test_get_job_result_job_not_found(mock_context, mock_job_queue, sample_job_id):
    """Test getting result for a non-existent job."""
    mock_job_queue.get_job_status.return_value = None

    result = await get_job_result(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    mock_job_queue.get_job_result.assert_not_called()
    assert result["success"] is False
    assert f"Job {sample_job_id} not found" in result["error"]

@pytest.mark.asyncio
async def test_get_job_result_queue_error(mock_context, mock_job_queue, sample_job_id):
    """Test getting result when the job queue raises an error."""
    mock_job_queue.get_job_status.return_value = {
        "job_id": sample_job_id,
        "status": "completed"
    }
    mock_job_queue.get_job_result.side_effect = Exception("Queue result failed")

    result = await get_job_result(job_id=sample_job_id, context=mock_context)

    assert result["success"] is False
    assert "Failed to get job result: Queue result failed" in result["error"]

# --- Tests for wait_for_result ---
@pytest.mark.asyncio
async def test_wait_for_result_success(mock_context, mock_job_queue, sample_job_id):
    """Test waiting for a job that completes successfully."""
    mock_job_queue.get_job_status.return_value = {"status": "pending"}
    mock_job_queue.wait_for_result.return_value = {
        "job_id": sample_job_id,
        "success": True,
        "result": "waited_output"
    }

    result = await wait_for_result(job_id=sample_job_id, timeout=1, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    mock_job_queue.wait_for_result.assert_called_once_with(sample_job_id, 1)
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert result["result"]["result"] == "waited_output"

@pytest.mark.asyncio
async def test_wait_for_result_timeout(mock_context, mock_job_queue, sample_job_id):
    """Test waiting for a job that times out."""
    status_response = {
        "job_id": sample_job_id,
        "status": "running",
        "created_at": 12345.678,
        "user_id": mock_context["user"]["id"]
    }
    mock_job_queue.get_job_status.return_value = status_response
    mock_job_queue.wait_for_result.return_value = None  # Simulate timeout

    result = await wait_for_result(job_id=sample_job_id, timeout=0.1, context=mock_context)

    mock_job_queue.get_job_status.assert_called_with(sample_job_id)  # Called twice
    mock_job_queue.wait_for_result.assert_called_once_with(sample_job_id, 0.1)
    assert result["success"] is False
    assert result["error"] == "Timeout waiting for job completion"
    assert result["status"] == status_response

@pytest.mark.asyncio
async def test_wait_for_result_job_not_found(mock_context, mock_job_queue, sample_job_id):
    """Test waiting for a non-existent job."""
    mock_job_queue.get_job_status.return_value = None

    result = await wait_for_result(job_id=sample_job_id, context=mock_context)

    mock_job_queue.get_job_status.assert_called_once_with(sample_job_id)
    mock_job_queue.wait_for_result.assert_not_called()
    assert result["success"] is False
    assert f"Job {sample_job_id} not found" in result["error"]

@pytest.mark.asyncio
async def test_wait_for_result_queue_error(mock_context, mock_job_queue, sample_job_id):
    """Test waiting for result when the job queue raises an error."""
    mock_job_queue.get_job_status.return_value = {"status": "pending"}
    mock_job_queue.wait_for_result.side_effect = Exception("Queue wait failed")

    result = await wait_for_result(job_id=sample_job_id, context=mock_context)

    assert result["success"] is False
    assert "Failed to wait for job result: Queue wait failed" in result["error"]

# --- Tests for list_jobs ---
@pytest.mark.asyncio
async def test_list_jobs_success(mock_context, mock_job_queue):
    """Test successful listing of jobs."""
    mock_jobs = [
        {
            "job_id": "job-1",
            "status": "completed",
            "user_id": mock_context["user"]["id"]
        },
        {
            "job_id": "job-2", 
            "status": "running",
            "user_id": mock_context["user"]["id"]
        }
    ]
    mock_job_queue.list_jobs.return_value = mock_jobs

    result = await list_jobs(context=mock_context)

    mock_job_queue.list_jobs.assert_called_once_with(
        mock_context["user"]["id"],  # Default to current user
        100,  # Default limit
        None   # Default status filter
    )
    assert result["success"] is True
    assert len(result["jobs"]) == 2
    assert result["jobs"] == mock_jobs

@pytest.mark.asyncio
async def test_list_jobs_with_filters(mock_context, mock_job_queue):
    """Test listing jobs with specific filters."""
    mock_jobs = [
        {
            "job_id": "job-1",
            "status": "completed",
            "user_id": "specific_user"
        }
    ]
    mock_job_queue.list_jobs.return_value = mock_jobs

    result = await list_jobs(
        user_id="specific_user",
        limit=50,
        status="completed",
        context=mock_context
    )

    mock_job_queue.list_jobs.assert_called_once_with(
        "specific_user",
        50,
        "completed"
    )
    assert result["success"] is True
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["status"] == "completed"

@pytest.mark.asyncio
async def test_list_jobs_empty(mock_context, mock_job_queue):
    """Test listing jobs when no jobs are found."""
    mock_job_queue.list_jobs.return_value = []

    result = await list_jobs(context=mock_context)

    assert result["success"] is True
    assert len(result["jobs"]) == 0

@pytest.mark.asyncio
async def test_list_jobs_queue_error(mock_context, mock_job_queue):
    """Test listing jobs when the job queue raises an error."""
    mock_job_queue.list_jobs.side_effect = Exception("Queue list failed")

    result = await list_jobs(context=mock_context)

    assert result["success"] is False
    assert "Failed to list jobs: Queue list failed" in result["error"]

# --- Tests for cancel_job ---
@pytest.mark.asyncio
async def test_cancel_job_success(mock_context, mock_job_queue, sample_job_id):
    """Test successful job cancellation."""
    mock_job_queue.cancel_job.return_value = {
        "success": True,
        "job_id": sample_job_id,
        "message": "Job cancelled successfully",
        "status": "cancelled"
    }

    result = await cancel_job(job_id=sample_job_id, context=mock_context)

    mock_job_queue.cancel_job.assert_called_once_with(
        sample_job_id,
        mock_context["user"]["id"]
    )
    assert result["success"] is True
    assert result["job_id"] == sample_job_id
    assert result["status"] == "cancelled"

@pytest.mark.asyncio
async def test_cancel_job_not_found(mock_context, mock_job_queue, sample_job_id):
    """Test cancelling a non-existent job."""
    mock_job_queue.cancel_job.return_value = {
        "success": False,
        "job_id": sample_job_id,
        "message": "Job not found",
        "status": None
    }

    result = await cancel_job(job_id=sample_job_id, context=mock_context)

    assert result["success"] is False
    assert result["job_id"] == sample_job_id
    assert "Job not found" in result["message"]

@pytest.mark.asyncio
async def test_cancel_job_unauthorized(mock_context, mock_job_queue, sample_job_id):
    """Test cancelling a job without proper authorization."""
    mock_job_queue.cancel_job.return_value = {
        "success": False,
        "job_id": sample_job_id,
        "message": "Unauthorized to cancel this job",
        "status": "running"
    }

    result = await cancel_job(job_id=sample_job_id, context=mock_context)

    assert result["success"] is False
    assert "Unauthorized" in result["message"]

@pytest.mark.asyncio
async def test_cancel_job_queue_error(mock_context, mock_job_queue, sample_job_id):
    """Test cancelling a job when the job queue raises an error."""
    mock_job_queue.cancel_job.side_effect = Exception("Queue cancel failed")

    result = await cancel_job(job_id=sample_job_id, context=mock_context)

    assert result["success"] is False
    assert "Failed to cancel job: Queue cancel failed" in result["message"] 