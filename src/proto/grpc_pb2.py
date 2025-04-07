# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpc.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ngrpc.proto\"/\n\x08MetaData\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x11\n\tfile_name\x18\x02 \x01(\t\"L\n\nUploadFile\x12\x1d\n\x08metadata\x18\x01 \x01(\x0b\x32\t.MetaDataH\x00\x12\x14\n\nchunk_data\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"\x1a\n\x04\x46ile\x12\x12\n\nchunk_data\x18\x01 \x01(\x0c\"\x1e\n\x0eStringResponse\x12\x0c\n\x04text\x18\x01 \x01(\t\"\x1b\n\x0b\x65\x63hoMessage\x12\x0c\n\x04text\x18\x01 \x01(\t\"\xab\x02\n\x10InitBenchRequest\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x13\n\x0bmodel_class\x18\x02 \x01(\t\x12\x14\n\x0cmodel_config\x18\x03 \x01(\x0c\x12\x12\n\ndataset_id\x18\x04 \x01(\t\x12\x12\n\nbatch_size\x18\x05 \x01(\x05\x12\x15\n\rlearning_rate\x18\x06 \x01(\x02\x12\x16\n\toptimizer\x18\x07 \x01(\x0cH\x01\x88\x01\x01\x12\x1a\n\rloss_function\x18\x08 \x01(\x0cH\x02\x88\x01\x01\x12\x1c\n\x12timeout_duration_s\x18\t \x01(\x02H\x00\x12\x1e\n\x14max_mini_batch_count\x18\n \x01(\x05H\x00\x42\t\n\x07requestB\x0c\n\n_optimizerB\x10\n\x0e_loss_function\"\xf9\x02\n\x10InitTrainRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x13\n\x0bmodel_class\x18\x03 \x01(\t\x12\x14\n\x0cmodel_config\x18\x04 \x01(\x0c\x12\x12\n\ndataset_id\x18\x05 \x01(\t\x12\x11\n\tmodel_wts\x18\x06 \x01(\x0c\x12\x12\n\nbatch_size\x18\x07 \x01(\x05\x12\x15\n\rlearning_rate\x18\x08 \x01(\x02\x12\x12\n\nnum_epochs\x18\t \x01(\x05\x12\x11\n\tround_idx\x18\n \x01(\x05\x12\x16\n\toptimizer\x18\x0b \x01(\x0cH\x01\x88\x01\x01\x12\x1a\n\rloss_function\x18\x0c \x01(\x0cH\x02\x88\x01\x01\x12\x1c\n\x12timeout_duration_s\x18\r \x01(\x02H\x00\x12\x1e\n\x14max_mini_batch_count\x18\x0e \x01(\x05H\x00\x42\t\n\x07requestB\x0c\n\n_optimizerB\x10\n\x0e_loss_function\"\x8a\x02\n\x15InitValidationRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x13\n\x0bmodel_class\x18\x03 \x01(\t\x12\x14\n\x0cmodel_config\x18\x04 \x01(\x0c\x12\x12\n\ndataset_id\x18\x05 \x01(\t\x12\x11\n\tmodel_wts\x18\x06 \x01(\x0c\x12\x12\n\nbatch_size\x18\x07 \x01(\x05\x12\x11\n\tround_idx\x18\x08 \x01(\x05\x12\x16\n\toptimizer\x18\t \x01(\x0cH\x00\x88\x01\x01\x12\x1a\n\rloss_function\x18\n \x01(\x0cH\x01\x88\x01\x01\x42\x0c\n\n_optimizerB\x10\n\x0e_loss_function\"Y\n\x11InitBenchResponse\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x18\n\x10num_mini_batches\x18\x02 \x01(\x05\x12\x18\n\x10\x62\x65nch_duration_s\x18\x03 \x01(\x02\"s\n\x11InitTrainResponse\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x15\n\rmodel_weights\x18\x02 \x01(\x0c\x12\x11\n\tclient_id\x18\x03 \x01(\t\x12\x11\n\tround_idx\x18\x04 \x01(\x05\x12\x0f\n\x07metrics\x18\x05 \x01(\x0c\"a\n\x16InitValidationResponse\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x11\n\tclient_id\x18\x02 \x01(\t\x12\x11\n\tround_idx\x18\x03 \x01(\x05\x12\x0f\n\x07metrics\x18\x04 \x01(\x0c\x32\x99\x02\n\x0b\x45\x64geService\x12$\n\x04\x45\x63ho\x12\x0c.echoMessage\x1a\x0c.echoMessage\"\x00\x12\x34\n\tInitBench\x12\x11.InitBenchRequest\x1a\x12.InitBenchResponse\"\x00\x12\x38\n\rStartTraining\x12\x11.InitTrainRequest\x1a\x12.InitTrainResponse\"\x00\x12.\n\nStreamFile\x12\x0b.UploadFile\x1a\x0f.StringResponse\"\x00(\x01\x12\x44\n\x0fStartValidation\x12\x16.InitValidationRequest\x1a\x17.InitValidationResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'grpc_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _METADATA._serialized_start=14
  _METADATA._serialized_end=61
  _UPLOADFILE._serialized_start=63
  _UPLOADFILE._serialized_end=139
  _FILE._serialized_start=141
  _FILE._serialized_end=167
  _STRINGRESPONSE._serialized_start=169
  _STRINGRESPONSE._serialized_end=199
  _ECHOMESSAGE._serialized_start=201
  _ECHOMESSAGE._serialized_end=228
  _INITBENCHREQUEST._serialized_start=231
  _INITBENCHREQUEST._serialized_end=530
  _INITTRAINREQUEST._serialized_start=533
  _INITTRAINREQUEST._serialized_end=910
  _INITVALIDATIONREQUEST._serialized_start=913
  _INITVALIDATIONREQUEST._serialized_end=1179
  _INITBENCHRESPONSE._serialized_start=1181
  _INITBENCHRESPONSE._serialized_end=1270
  _INITTRAINRESPONSE._serialized_start=1272
  _INITTRAINRESPONSE._serialized_end=1387
  _INITVALIDATIONRESPONSE._serialized_start=1389
  _INITVALIDATIONRESPONSE._serialized_end=1486
  _EDGESERVICE._serialized_start=1489
  _EDGESERVICE._serialized_end=1770
# @@protoc_insertion_point(module_scope)
