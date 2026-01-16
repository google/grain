.. currentmodule:: grain

``grain`` package
=============================

.. toctree::
   :hidden:
   :caption: Core APIs

   Dataset APIs <grain.dataset>
   DataLoader APIs <grain.data_loader>

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
----------------------------

.. autosummary::

   MapDataset
   IterDataset
   DatasetIterator
   ReadOptions


Simple high-level pipelines
---------------------------


.. autosummary::

   load
   DataLoader
   DataLoaderIterator
   Record
   RecordMetadata