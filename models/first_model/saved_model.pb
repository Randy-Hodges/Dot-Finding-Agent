ù»
®
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ÀÅ

while/actor_critic/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*0
shared_name!while/actor_critic/dense/kernel

3while/actor_critic/dense/kernel/Read/ReadVariableOpReadVariableOpwhile/actor_critic/dense/kernel*
_output_shapes
:	
*
dtype0

while/actor_critic/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namewhile/actor_critic/dense/bias

1while/actor_critic/dense/bias/Read/ReadVariableOpReadVariableOpwhile/actor_critic/dense/bias*
_output_shapes	
:*
dtype0
 
!while/actor_critic/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!while/actor_critic/dense_1/kernel

5while/actor_critic/dense_1/kernel/Read/ReadVariableOpReadVariableOp!while/actor_critic/dense_1/kernel* 
_output_shapes
:
*
dtype0

while/actor_critic/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!while/actor_critic/dense_1/bias

3while/actor_critic/dense_1/bias/Read/ReadVariableOpReadVariableOpwhile/actor_critic/dense_1/bias*
_output_shapes	
:*
dtype0

!while/actor_critic/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*2
shared_name#!while/actor_critic/dense_2/kernel

5while/actor_critic/dense_2/kernel/Read/ReadVariableOpReadVariableOp!while/actor_critic/dense_2/kernel*
_output_shapes
:		*
dtype0

while/actor_critic/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!while/actor_critic/dense_2/bias

3while/actor_critic/dense_2/bias/Read/ReadVariableOpReadVariableOpwhile/actor_critic/dense_2/bias*
_output_shapes
:	*
dtype0

!while/actor_critic/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!while/actor_critic/dense_3/kernel

5while/actor_critic/dense_3/kernel/Read/ReadVariableOpReadVariableOp!while/actor_critic/dense_3/kernel*
_output_shapes
:	*
dtype0

while/actor_critic/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!while/actor_critic/dense_3/bias

3while/actor_critic/dense_3/bias/Read/ReadVariableOpReadVariableOpwhile/actor_critic/dense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
È
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueùBö Bï


common
common2
	actor

critic
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
h


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
8

0
1
2
3
4
5
6
7
 
8

0
1
2
3
4
5
6
7
­
	variables
"metrics
#layer_regularization_losses
$non_trainable_variables

%layers
&layer_metrics
regularization_losses
trainable_variables
 
][
VARIABLE_VALUEwhile/actor_critic/dense/kernel(common/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEwhile/actor_critic/dense/bias&common/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
­
	variables
'metrics
(layer_regularization_losses

)layers
*non_trainable_variables
+layer_metrics
regularization_losses
trainable_variables
`^
VARIABLE_VALUE!while/actor_critic/dense_1/kernel)common2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEwhile/actor_critic/dense_1/bias'common2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
,metrics
-layer_regularization_losses

.layers
/non_trainable_variables
0layer_metrics
regularization_losses
trainable_variables
^\
VARIABLE_VALUE!while/actor_critic/dense_2/kernel'actor/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEwhile/actor_critic/dense_2/bias%actor/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
1metrics
2layer_regularization_losses

3layers
4non_trainable_variables
5layer_metrics
regularization_losses
trainable_variables
_]
VARIABLE_VALUE!while/actor_critic/dense_3/kernel(critic/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEwhile/actor_critic/dense_3/bias&critic/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
6metrics
7layer_regularization_losses

8layers
9non_trainable_variables
:layer_metrics
regularization_losses
 trainable_variables
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1while/actor_critic/dense/kernelwhile/actor_critic/dense/bias!while/actor_critic/dense_1/kernelwhile/actor_critic/dense_1/bias!while/actor_critic/dense_2/kernelwhile/actor_critic/dense_2/bias!while/actor_critic/dense_3/kernelwhile/actor_critic/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_363243
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3while/actor_critic/dense/kernel/Read/ReadVariableOp1while/actor_critic/dense/bias/Read/ReadVariableOp5while/actor_critic/dense_1/kernel/Read/ReadVariableOp3while/actor_critic/dense_1/bias/Read/ReadVariableOp5while/actor_critic/dense_2/kernel/Read/ReadVariableOp3while/actor_critic/dense_2/bias/Read/ReadVariableOp5while/actor_critic/dense_3/kernel/Read/ReadVariableOp3while/actor_critic/dense_3/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_363369
ª
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamewhile/actor_critic/dense/kernelwhile/actor_critic/dense/bias!while/actor_critic/dense_1/kernelwhile/actor_critic/dense_1/bias!while/actor_critic/dense_2/kernelwhile/actor_critic/dense_2/bias!while/actor_critic/dense_3/kernelwhile/actor_critic/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_363403
ñ	
Ú
A__inference_dense_layer_call_and_return_conditional_losses_363254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ø
{
&__inference_dense_layer_call_fn_363263

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3630972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Ü
C__inference_dense_3_layer_call_and_return_conditional_losses_363312

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
Ü
C__inference_dense_1_layer_call_and_return_conditional_losses_363124

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_dense_2_layer_call_and_return_conditional_losses_363293

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô	
ä
$__inference_signature_wrapper_363243
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_3630822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
	
Ü
C__inference_dense_3_layer_call_and_return_conditional_losses_363176

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
Ú
A__inference_dense_layer_call_and_return_conditional_losses_363097

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ü
}
(__inference_dense_3_layer_call_fn_363321

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3631762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
'
°
"__inference__traced_restore_363403
file_prefix4
0assignvariableop_while_actor_critic_dense_kernel4
0assignvariableop_1_while_actor_critic_dense_bias8
4assignvariableop_2_while_actor_critic_dense_1_kernel6
2assignvariableop_3_while_actor_critic_dense_1_bias8
4assignvariableop_4_while_actor_critic_dense_2_kernel6
2assignvariableop_5_while_actor_critic_dense_2_bias8
4assignvariableop_6_while_actor_critic_dense_3_kernel6
2assignvariableop_7_while_actor_critic_dense_3_bias

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7ï
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*û
valueñBî	B(common/kernel/.ATTRIBUTES/VARIABLE_VALUEB&common/bias/.ATTRIBUTES/VARIABLE_VALUEB)common2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'common2/bias/.ATTRIBUTES/VARIABLE_VALUEB'actor/kernel/.ATTRIBUTES/VARIABLE_VALUEB%actor/bias/.ATTRIBUTES/VARIABLE_VALUEB(critic/kernel/.ATTRIBUTES/VARIABLE_VALUEB&critic/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesØ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¯
AssignVariableOpAssignVariableOp0assignvariableop_while_actor_critic_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1µ
AssignVariableOp_1AssignVariableOp0assignvariableop_1_while_actor_critic_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¹
AssignVariableOp_2AssignVariableOp4assignvariableop_2_while_actor_critic_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3·
AssignVariableOp_3AssignVariableOp2assignvariableop_3_while_actor_critic_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¹
AssignVariableOp_4AssignVariableOp4assignvariableop_4_while_actor_critic_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5·
AssignVariableOp_5AssignVariableOp2assignvariableop_5_while_actor_critic_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¹
AssignVariableOp_6AssignVariableOp4assignvariableop_6_while_actor_critic_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7·
AssignVariableOp_7AssignVariableOp2assignvariableop_7_while_actor_critic_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ö	
Ü
C__inference_dense_1_layer_call_and_return_conditional_losses_363274

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§2

!__inference__wrapped_model_363082
input_15
1actor_critic_dense_matmul_readvariableop_resource6
2actor_critic_dense_biasadd_readvariableop_resource7
3actor_critic_dense_1_matmul_readvariableop_resource8
4actor_critic_dense_1_biasadd_readvariableop_resource7
3actor_critic_dense_2_matmul_readvariableop_resource8
4actor_critic_dense_2_biasadd_readvariableop_resource7
3actor_critic_dense_3_matmul_readvariableop_resource8
4actor_critic_dense_3_biasadd_readvariableop_resource
identity

identity_1¢)actor_critic/dense/BiasAdd/ReadVariableOp¢(actor_critic/dense/MatMul/ReadVariableOp¢+actor_critic/dense_1/BiasAdd/ReadVariableOp¢*actor_critic/dense_1/MatMul/ReadVariableOp¢+actor_critic/dense_2/BiasAdd/ReadVariableOp¢*actor_critic/dense_2/MatMul/ReadVariableOp¢+actor_critic/dense_3/BiasAdd/ReadVariableOp¢*actor_critic/dense_3/MatMul/ReadVariableOpÇ
(actor_critic/dense/MatMul/ReadVariableOpReadVariableOp1actor_critic_dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02*
(actor_critic/dense/MatMul/ReadVariableOp®
actor_critic/dense/MatMulMatMulinput_10actor_critic/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense/MatMulÆ
)actor_critic/dense/BiasAdd/ReadVariableOpReadVariableOp2actor_critic_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)actor_critic/dense/BiasAdd/ReadVariableOpÎ
actor_critic/dense/BiasAddBiasAdd#actor_critic/dense/MatMul:product:01actor_critic/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense/BiasAdd
actor_critic/dense/ReluRelu#actor_critic/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense/ReluÎ
*actor_critic/dense_1/MatMul/ReadVariableOpReadVariableOp3actor_critic_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*actor_critic/dense_1/MatMul/ReadVariableOpÒ
actor_critic/dense_1/MatMulMatMul%actor_critic/dense/Relu:activations:02actor_critic/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense_1/MatMulÌ
+actor_critic/dense_1/BiasAdd/ReadVariableOpReadVariableOp4actor_critic_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+actor_critic/dense_1/BiasAdd/ReadVariableOpÖ
actor_critic/dense_1/BiasAddBiasAdd%actor_critic/dense_1/MatMul:product:03actor_critic/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense_1/BiasAdd
actor_critic/dense_1/ReluRelu%actor_critic/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense_1/ReluÍ
*actor_critic/dense_2/MatMul/ReadVariableOpReadVariableOp3actor_critic_dense_2_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02,
*actor_critic/dense_2/MatMul/ReadVariableOpÓ
actor_critic/dense_2/MatMulMatMul'actor_critic/dense_1/Relu:activations:02actor_critic/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
actor_critic/dense_2/MatMulË
+actor_critic/dense_2/BiasAdd/ReadVariableOpReadVariableOp4actor_critic_dense_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02-
+actor_critic/dense_2/BiasAdd/ReadVariableOpÕ
actor_critic/dense_2/BiasAddBiasAdd%actor_critic/dense_2/MatMul:product:03actor_critic/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
actor_critic/dense_2/BiasAddÍ
*actor_critic/dense_3/MatMul/ReadVariableOpReadVariableOp3actor_critic_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*actor_critic/dense_3/MatMul/ReadVariableOpÓ
actor_critic/dense_3/MatMulMatMul'actor_critic/dense_1/Relu:activations:02actor_critic/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense_3/MatMulË
+actor_critic/dense_3/BiasAdd/ReadVariableOpReadVariableOp4actor_critic_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+actor_critic/dense_3/BiasAdd/ReadVariableOpÕ
actor_critic/dense_3/BiasAddBiasAdd%actor_critic/dense_3/MatMul:product:03actor_critic/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
actor_critic/dense_3/BiasAddá
IdentityIdentity%actor_critic/dense_2/BiasAdd:output:0*^actor_critic/dense/BiasAdd/ReadVariableOp)^actor_critic/dense/MatMul/ReadVariableOp,^actor_critic/dense_1/BiasAdd/ReadVariableOp+^actor_critic/dense_1/MatMul/ReadVariableOp,^actor_critic/dense_2/BiasAdd/ReadVariableOp+^actor_critic/dense_2/MatMul/ReadVariableOp,^actor_critic/dense_3/BiasAdd/ReadVariableOp+^actor_critic/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identityå

Identity_1Identity%actor_critic/dense_3/BiasAdd:output:0*^actor_critic/dense/BiasAdd/ReadVariableOp)^actor_critic/dense/MatMul/ReadVariableOp,^actor_critic/dense_1/BiasAdd/ReadVariableOp+^actor_critic/dense_1/MatMul/ReadVariableOp,^actor_critic/dense_2/BiasAdd/ReadVariableOp+^actor_critic/dense_2/MatMul/ReadVariableOp,^actor_critic/dense_3/BiasAdd/ReadVariableOp+^actor_critic/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ
::::::::2V
)actor_critic/dense/BiasAdd/ReadVariableOp)actor_critic/dense/BiasAdd/ReadVariableOp2T
(actor_critic/dense/MatMul/ReadVariableOp(actor_critic/dense/MatMul/ReadVariableOp2Z
+actor_critic/dense_1/BiasAdd/ReadVariableOp+actor_critic/dense_1/BiasAdd/ReadVariableOp2X
*actor_critic/dense_1/MatMul/ReadVariableOp*actor_critic/dense_1/MatMul/ReadVariableOp2Z
+actor_critic/dense_2/BiasAdd/ReadVariableOp+actor_critic/dense_2/BiasAdd/ReadVariableOp2X
*actor_critic/dense_2/MatMul/ReadVariableOp*actor_critic/dense_2/MatMul/ReadVariableOp2Z
+actor_critic/dense_3/BiasAdd/ReadVariableOp+actor_critic/dense_3/BiasAdd/ReadVariableOp2X
*actor_critic/dense_3/MatMul/ReadVariableOp*actor_critic/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
	
Ü
C__inference_dense_2_layer_call_and_return_conditional_losses_363150

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
}
(__inference_dense_2_layer_call_fn_363302

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3631502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
-__inference_actor_critic_layer_call_fn_363218
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_actor_critic_layer_call_and_return_conditional_losses_3631942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
¸

H__inference_actor_critic_layer_call_and_return_conditional_losses_363194
input_1
dense_363108
dense_363110
dense_1_363135
dense_1_363137
dense_2_363161
dense_2_363163
dense_3_363187
dense_3_363189
identity

identity_1¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_363108dense_363110*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3630972
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_363135dense_1_363137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3631242!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_363161dense_2_363163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3631502!
dense_2/StatefulPartitionedCall±
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_363187dense_3_363189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3631762!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ
::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Þ
}
(__inference_dense_1_layer_call_fn_363283

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3631242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
ð
__inference__traced_save_363369
file_prefix>
:savev2_while_actor_critic_dense_kernel_read_readvariableop<
8savev2_while_actor_critic_dense_bias_read_readvariableop@
<savev2_while_actor_critic_dense_1_kernel_read_readvariableop>
:savev2_while_actor_critic_dense_1_bias_read_readvariableop@
<savev2_while_actor_critic_dense_2_kernel_read_readvariableop>
:savev2_while_actor_critic_dense_2_bias_read_readvariableop@
<savev2_while_actor_critic_dense_3_kernel_read_readvariableop>
:savev2_while_actor_critic_dense_3_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameé
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*û
valueñBî	B(common/kernel/.ATTRIBUTES/VARIABLE_VALUEB&common/bias/.ATTRIBUTES/VARIABLE_VALUEB)common2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'common2/bias/.ATTRIBUTES/VARIABLE_VALUEB'actor/kernel/.ATTRIBUTES/VARIABLE_VALUEB%actor/bias/.ATTRIBUTES/VARIABLE_VALUEB(critic/kernel/.ATTRIBUTES/VARIABLE_VALUEB&critic/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices¦
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_while_actor_critic_dense_kernel_read_readvariableop8savev2_while_actor_critic_dense_bias_read_readvariableop<savev2_while_actor_critic_dense_1_kernel_read_readvariableop:savev2_while_actor_critic_dense_1_bias_read_readvariableop<savev2_while_actor_critic_dense_2_kernel_read_readvariableop:savev2_while_actor_critic_dense_2_bias_read_readvariableop<savev2_while_actor_critic_dense_3_kernel_read_readvariableop:savev2_while_actor_critic_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*^
_input_shapesM
K: :	
::
::		:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:		: 

_output_shapes
:	:%!

_output_shapes
:	: 

_output_shapes
::	

_output_shapes
: "±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*é
serving_defaultÕ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ	<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:æe
ð

common
common2
	actor

critic
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
*;&call_and_return_all_conditional_losses
<_default_save_signature
=__call__"
_tf_keras_modelê{"class_name": "ActorCritic", "name": "actor_critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ActorCritic"}}
ê


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*>&call_and_return_all_conditional_losses
?__call__"Å
_tf_keras_layer«{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 10]}}
ð

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
ð

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
ð

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
*D&call_and_return_all_conditional_losses
E__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
Ê
	variables
"metrics
#layer_regularization_losses
$non_trainable_variables

%layers
&layer_metrics
regularization_losses
trainable_variables
=__call__
<_default_save_signature
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
,
Fserving_default"
signature_map
2:0	
2while/actor_critic/dense/kernel
,:*2while/actor_critic/dense/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­
	variables
'metrics
(layer_regularization_losses

)layers
*non_trainable_variables
+layer_metrics
regularization_losses
trainable_variables
?__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
5:3
2!while/actor_critic/dense_1/kernel
.:,2while/actor_critic/dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
,metrics
-layer_regularization_losses

.layers
/non_trainable_variables
0layer_metrics
regularization_losses
trainable_variables
A__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
4:2		2!while/actor_critic/dense_2/kernel
-:+	2while/actor_critic/dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
1metrics
2layer_regularization_losses

3layers
4non_trainable_variables
5layer_metrics
regularization_losses
trainable_variables
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
4:2	2!while/actor_critic/dense_3/kernel
-:+2while/actor_critic/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
6metrics
7layer_regularization_losses

8layers
9non_trainable_variables
:layer_metrics
regularization_losses
 trainable_variables
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
2
H__inference_actor_critic_layer_call_and_return_conditional_losses_363194Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ß2Ü
!__inference__wrapped_model_363082¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

û2ø
-__inference_actor_critic_layer_call_fn_363218Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ë2è
A__inference_dense_layer_call_and_return_conditional_losses_363254¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_363263¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_363274¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_363283¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_363293¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_363302¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_363312¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_363321¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_363243input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ç
!__inference__wrapped_model_363082¡
0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ	
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿÖ
H__inference_actor_critic_layer_call_and_return_conditional_losses_363194
0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 ¬
-__inference_actor_critic_layer_call_fn_363218{
0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "=¢:

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_1_layer_call_and_return_conditional_losses_363274^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_1_layer_call_fn_363283Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_2_layer_call_and_return_conditional_losses_363293]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ	
 |
(__inference_dense_2_layer_call_fn_363302P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ	¤
C__inference_dense_3_layer_call_and_return_conditional_losses_363312]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_3_layer_call_fn_363321P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
A__inference_dense_layer_call_and_return_conditional_losses_363254]
/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
&__inference_dense_layer_call_fn_363263P
/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿÕ
$__inference_signature_wrapper_363243¬
;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
"cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ	
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ