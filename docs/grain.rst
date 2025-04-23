.. currentmodule:: grain

Public API: ``grain`` package
===========================

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   grain.checkpoint
   grain.constants
   grain.experimental
   grain.multiprocessing
   grain.samplers
   grain.sharding
   grain.sources
   grain.transforms


Flexible low-level pipelines
-------------

.. autosummary::

   MapDataset
   IterDataset
   DatasetIterator
   ReadOptions


Simple high-level pipelines
-------------


.. autosummary::
   :toctree: _autosummary

   load
   DataLoader
   DataLoaderIterator
   Record
   RecordMetadata