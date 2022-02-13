import torch

x = torch.randn(3, requires_grad=True)
print(x)

# This is considered a forward pass, where inputs are x and 2 and output is y
y = x + 2
print(y)

# Basically torch adds a function that will be use to calculate
# the gradient grad_fn=<AddBackward0>

# Notice how the type of the grad_fn changes as we apply operations
z = y * y * 2
print(z)  # tensor([ 3.0578, 10.3792, 14.9571], grad_fn=<MulBackward0>)
z = z.mean()
print(z)  # tensor(9.4647, grad_fn=<MeanBackward0>)

# since z is a scalar we do not need a parameter in backward
z.backward() # dz/dx
print(x.grad)

print()
a = torch.randn(3, requires_grad=True)
print(a)
b = a * a

# Since b is not a scalar we need to provide a vector
# Internally if the vector that is multiplied by the Jacobian matrix
v = torch.tensor([0.1, 1.0, 0.001])
b.backward(v)
print(a.grad)

#########
# Note that operations with tensors that require grad are tracked.
# If we want to do some operations that are not part of the gradients:

x = torch.randn(3, requires_grad=True)
print(x)

#x.requires_grad_(False)
#x.detach()

with torch.no_grad():
    # w is not considered a transformation used by the gradient
    w = x / 2
    print("w does not have grad_fn")
    print("w = ", w)

##########

weights = torch.ones(4, requires_grad=True)

# In torch, instead of all this manual code we use optimizers

for epoch in range(10):
    print("weights=", weights)
    model_output = (weights*3).sum()
    model_output.backward()

    print("weights.grad=", weights.grad)

    with torch.no_grad():
        # We optimize our weights which consists on
        # moving them using the grad and a learning rate
        # this is how SGD would do it
        weights_ = weights - 0.01 * weights.grad
        # we need to change the weights in place
        weights.copy_(weights_)

    # Before the next optimization step we must set the gradients to 0
    # otherwise those accumulate
    weights.grad.zero_()



