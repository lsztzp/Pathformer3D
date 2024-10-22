work_dir = ""

image_input_resize = (128, 256)
action_map_size = (128, 256)

feature_dim = 448  # 192 384 448 512 576
patch_size = (8, 8)

num_patch_h = image_input_resize[0] // patch_size[0]
num_patch_w = image_input_resize[1] // patch_size[1]
num_patchs = num_patch_h * num_patch_w

max_length = 30

# --------------------------------------------------------------------------------------------
#  MODEL PARAMETER
# --------------------------------------------------------------------------------------------
d_model = 128  # Embedding Size
d_ff = 128  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
enc_n_layers = 4  # number of Encoder of Decoder Layer
dec_n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
dropout = 0
num_gauss = 5
MDN_hidden_num = 16
postion_method = 'fixed'

# --------------------------------------------------------------------------------------------
#  Train Setting
# --------------------------------------------------------------------------------------------
device = "cuda:0"
lr = 5e-6
epoch_nums = 40
train_batch_size = 18
val_batch_size = 2

val_step = 1
seed = 1218

weight_decay = 0
lr_scheduler = dict(
    type='MultiStepLR',
    warmup_epochs=10,
    milestones=[10, 20, 30, 40],
    gamma=0.5)

feature_grad = True
reload_path = ""

# --------------------------------------------------------------------------------------------
#  Test Setting
# --------------------------------------------------------------------------------------------
sphere_constraint_loss = False
replace_encoder = False

train_dataset = 'merge'