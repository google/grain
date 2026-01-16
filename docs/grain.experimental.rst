``grain.experimental`` module
=============================

.. automodule:: grain.experimental

List of Members
---------------

.. autoclass:: FlatMapTransform
   :members:

.. autoclass:: DatasetOptions
   :members:

.. autoclass:: ExecutionTrackingMode
   :members:

.. autofunction:: apply_transformations

.. autoclass:: ElasticIterator
   :special-members: __init__, __iter__, __next__

.. autoclass:: WithOptionsIterDataset
   :special-members: __init__, __iter__

.. autoclass:: ParquetIterDataset
   :special-members: __init__, __iter__

.. autoclass:: TFRecordIterDataset
   :special-members: __init__, __iter__

.. autofunction:: batch_and_pad

.. autoclass:: CacheIterDataset
   :special-members: __init__, __iter__

.. autoclass:: FlatMapMapDataset
   :special-members: __init__, __getitem__

.. autoclass:: FlatMapIterDataset
   :special-members: __init__, __iter__

.. autoclass:: InterleaveIterDataset
   :special-members: __init__, __iter__

.. autoclass:: LimitIterDataset
   :special-members: __init__, __iter__

.. autoclass:: RngPool
   :members:

.. autoclass:: FirstFitPackIterDataset
   :special-members: __init__, __iter__

.. autoclass:: BestFitPackIterDataset
   :special-members: __init__, __iter__

.. autoclass:: BOSHandling
   :members:

.. autoclass:: ConcatThenSplitIterDataset
   :special-members: __init__, __iter__

.. autofunction:: multithread_prefetch

.. autoclass:: ThreadPrefetchIterDataset
   :special-members: __init__, __iter__

.. autoclass:: ThreadPrefetchDatasetIterator
   :special-members: __init__, __iter__, __next__

.. autoclass:: RebatchIterDataset
   :special-members: __init__, __iter__

.. autoclass:: RepeatIterDataset
   :special-members: __init__, __iter__


.. autoclass:: WindowShuffleMapDataset
   :special-members: __init__, __getitem__

.. autoclass:: WindowShuffleIterDataset
   :special-members: __init__, __iter__


.. autoclass:: ZipMapDataset
   :special-members: __init__, __getitem__


.. autoclass:: ZipIterDataset
   :special-members: __init__, __iter__

.. autofunction:: index_shuffle

.. autofunction:: assert_equal_output_after_checkpoint

.. autofunction:: device_put

.. autoclass:: PerformanceConfig
   :members:

.. autofunction:: pick_performance_config

.. autofunction:: get_element_spec

.. autofunction:: set_next_index

.. autofunction:: get_next_index
