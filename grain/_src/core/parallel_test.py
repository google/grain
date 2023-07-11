"""Tests for parallel."""
import threading

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import parallel


def ReturnThreadName():
  return threading.current_thread().name


def Identity(i):
  return i


def FnThatAlwaysFails(arg):
  del arg
  raise ValueError("I always fail")


def FnThatFailsOnOddInputs(i):
  if i % 2 == 1:
    raise ValueError("Failed on an odd input")
  return i


class ParallelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name=" empty list of args",
          num_workers=1,
          input_dict_list=[],
          expected=[],
      ),
      dict(
          testcase_name=" one worker, nonempty list",
          num_workers=1,
          input_dict_list=[dict(i=k) for k in range(1, 10)],
          expected=list(range(1, 10)),
      ),
      dict(
          testcase_name=" fewer workers than jobs, nonempty list",
          num_workers=3,
          input_dict_list=[dict(i=k) for k in range(1, 10)],
          expected=list(range(1, 10)),
      ),
      dict(
          testcase_name=" more workers than jobs, nonempty list",
          num_workers=20,
          input_dict_list=[dict(i=k) for k in range(1, 10)],
          expected=list(range(1, 10)),
      ),
  )
  def testRunInParallel(self, input_dict_list, num_workers: int, expected):
    actual = parallel.run_in_parallel(Identity, input_dict_list, num_workers)
    self.assertEqual(actual, expected)

  def testRunInParallelOnAlwaysFailingFn(self):
    with self.assertRaisesRegex(ValueError, "I always fail"):
      parallel.run_in_parallel(FnThatAlwaysFails, [dict(arg="hi")], 10)

  @parameterized.named_parameters(
      dict(
          testcase_name=" one failing input, one worker",
          num_workers=1,
          input_dict_list=[{"i": 1}],
      ),
      dict(
          testcase_name=" one failing input, many workers",
          num_workers=5,
          input_dict_list=[{"i": 1}],
      ),
      dict(
          testcase_name=" one failing input, one succeeding input",
          num_workers=5,
          input_dict_list=[{"i": 1}, {"i": 2}],
      ),
      dict(
          testcase_name=" two failing inputs, one succeeding input",
          num_workers=5,
          input_dict_list=[{"i": 1}, {"i": 2}, {"i": 3}],
      ),
  )
  def testRunInParallelFailsIfSomeFnCallsFail(
      self, input_dict_list, num_workers: int
  ):
    with self.assertRaisesRegex(ValueError, "Failed on an odd input"):
      parallel.run_in_parallel(
          FnThatFailsOnOddInputs, input_dict_list, num_workers
      )

  def testRunInParallelForThreadNamePrefix(self):
    input_kwarg_list = [{}]
    thread_names = parallel.run_in_parallel(
        ReturnThreadName, input_kwarg_list, 5, thread_name_prefix="Customized-"
    )
    self.assertStartsWith(thread_names[0], "Customized-")

  def testRunInParallelForDefaultThreadNamePrefix(self):
    input_kwarg_list = [{}]
    thread_names = parallel.run_in_parallel(
        ReturnThreadName, input_kwarg_list, 5
    )
    self.assertStartsWith(thread_names[0], "parallel_")


if __name__ == "__main__":
  absltest.main()
