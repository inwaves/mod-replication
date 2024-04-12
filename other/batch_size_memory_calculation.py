# src: https://www.trentonbricken.com/TransformerMemoryRequirements/

L = 12 # number of blocks
N = 12 # number of heads
E = 768 # embedding dimension
B = 11 # batch size
T = 1024 # sequence length
TOKS = 50257 # number of tokens in the vocab
param_bytes = 4 # float32 uses 4 bytes
bytes_to_gigs = 1_000_000_000 # 1 billion bytes in a gigabyte

model_params = (TOKS*E)+ L*( 4*E**2 + 2*E*4*E + 4*E)
act_params = B*T*(2*TOKS+L*(14*E + N*T ))
backprop_model_params = 3*model_params
backprop_act_params = act_params
total_params = model_params+act_params+backprop_model_params+backprop_act_params
prop_act_params=4*model_params+2*act_params
gigabytes_used = total_params*param_bytes/bytes_to_gigs

print(f"GiB used: {gigabytes_used:.2f}")
