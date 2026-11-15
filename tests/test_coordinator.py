import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from hivemind.coordinator import Coordinator

@pytest.fixture
def mock_env():
    """Sets up the coordinator with mocked dependencies."""
    store = MagicMock()
    memory = MagicMock()
    reply = AsyncMock()
    nodes = MagicMock(return_value=[{'id': 'node-1', 'capabilities': ['browser.open', 'files.read']}])

    coordinator = Coordinator(store, memory, reply, nodes)
    # Mock the create method so tests don't actually schedule new async tasks
    coordinator.create = MagicMock()
    return coordinator, store, reply

@pytest.mark.asyncio
async def test_advance_waits_for_pending_children(mock_env):
    coordinator, store, _ = mock_env
    task = {'id': 'task-1', 'prompt': 'Do work', 'node_id': None}

    # Simulate one completed child and one running child
    store.rows.return_value = [{'id': 'child-1'}, {'id': 'child-2'}]
    store.task.side_effect = [
        {'id': 'child-1', 'status': 'completed'},
        {'id': 'child-2', 'status': 'running'}
    ]

    await coordinator.advance(task)

    # Verify it returned early without planning
    store.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_advance_handles_handoff(mock_env):
    coordinator, store, reply = mock_env
    task = {'id': 'task-1', 'prompt': 'Ask user for input', 'node_id': None}

    store.rows.return_value = [] # No children yet

    # FIX: Tell the mock that the task is still in 'planning' state after LLM replies
    store.task.return_value = {'status': 'planning'}

    # Simulate LLM returning a handoff instruction
    reply.return_value = json.dumps({
        "handoff": "human_user",
        "message": "I need the API key to proceed."
    })

    await coordinator.advance(task)

    # Verify the task was completed and the message was saved
    store.update_task.assert_called_with(
        'task-1',
        'completed',
        result={'handoff': 'human_user', 'message': 'I need the API key to proceed.'}
    )
    store.message.assert_called_with('assistant', 'Handed off to human_user: I need the API key to proceed.', task_id='task-1')

@pytest.mark.asyncio
async def test_advance_dispatches_concurrent_actions(mock_env):
    coordinator, store, reply = mock_env
    task = {'id': 'task-1', 'prompt': 'Read two files', 'node_id': 'node-1'}

    store.rows.return_value = []

    # FIX: Tell the mock that the task is still in 'planning' state after LLM replies
    store.task.return_value = {'status': 'planning'}

    # Simulate LLM returning an array of two concurrent actions
    reply.return_value = json.dumps({
        "actions": [
            {"capability": "files.read", "node_id": "node-1", "arguments": {"path": "file1.txt"}},
            {"capability": "files.read", "node_id": "node-1", "arguments": {"path": "file2.txt"}}
        ]
    })

    await coordinator.advance(task)

    # Verify task state was updated to wait for children
    store.update_task.assert_any_call('task-1', 'waiting_children')

    # Verify self.create was called exactly twice (once for each action)
    assert coordinator.create.call_count == 2

    # Check the arguments passed to the second create call (step_no should be 2)
    coordinator.create.assert_called_with(
        'Read two files', # Falls back to task prompt if no reason provided
        'node-1',
        'files.read',
        {'path': 'file2.txt'},
        'task-1',
        2 # Step number
    )
