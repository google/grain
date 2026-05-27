from multiprocessing import shared_memory
import os
from unittest import mock
import uuid

from absl.testing import absltest
import multiprocessing as grain_mp


class MultiprocessingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(grain_mp, "_endoscope_shm_name", new=None)
    )

  def test_get_shm_name(self):
    fixed_uuid = "12345678-1234-5678-1234-567812345678"
    with mock.patch.object(uuid, "uuid4", autospec=True) as mock_uuid:
      mock_uuid.return_value = uuid.UUID(fixed_uuid)
      expected_name = (
          f"grain_endoscope_ports_shm_{os.getpid()}_{uuid.UUID(fixed_uuid).hex}"
      )
      name = grain_mp._get_shm_name()
      self.assertEqual(name, expected_name)
      self.assertIn(str(os.getpid()), name)

  def test_initialize_endoscope_shm_size_mismatch(self):
    test_name = f"test_grain_shm_{os.getpid()}"
    # Override the patch from setUp for this test.
    self.enter_context(
        mock.patch.object(grain_mp, "_endoscope_shm_name", new=test_name)
    )

    mock_shm = mock.create_autospec(shared_memory.SharedMemory, instance=True)
    mock_shm.size = 4  # Small size

    with mock.patch.object(
        shared_memory, "SharedMemory", autospec=True
    ) as mock_sm:
      mock_sm.return_value = mock_shm

      with self.assertRaisesRegex(
          ValueError, "Shared memory segment size mismatch: 4 != "
      ):
        grain_mp._initialize_endoscope_shm()


if __name__ == "__main__":
  grain_absltest.main()
