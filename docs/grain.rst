.. currentmodule:: grain

Public API: ``grain`` package
===========================

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   grain.checkpoint
   grain.constatnts
   grain.experimental
   grain.multiprocessing
   grain.samplers
   grain.sharding
   grain.sources
   grain.transforms

Dataset
-------------

   grain._src.python.dataset.dataset.MapDatasetMeta
   MapDataset
   grain._src.python.dataset.dataset.IterDatasetMeta
   IterDataset
   DatasetIterator

Dataset APIs
-------------

.. toctree::
   :maxdepth: 1

   grain.dataset

Higher level pipeline APIs
-------------


.. autosummary::
   :toctree: _autosummary

   load
   DataLoader
   DataLoaderIterator
   Record
   RecordMetadata