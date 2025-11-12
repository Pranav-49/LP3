# Q: Implement Gradient Descent Algorithm to find local minima of y = (x + 3)^2

# ---------------- Step 1: Import Library ----------------
import matplotlib.pyplot as plt

# ---------------- Step 2: Define Function and Derivative ----------------
def function(x):
    return (x + 3)**2   # y = (x + 3)^2

def derivative(x):
    return 2 * (x + 3)  # dy/dx = 2(x + 3)

# ---------------- Step 3: Gradient Descent Parameters ----------------
x = 2.0            # Starting point
learning_rate = 0.1
epochs = 50        # Number of iterations

# ---------------- Step 4: Perform Gradient Descent ----------------
x_values = [x]
for i in range(epochs):
    grad = derivative(x)
    x = x - learning_rate * grad
    x_values.append(x)
    print(f"Iteration {i+1}: x = {x:.5f}, f(x) = {function(x):.5f}")

# ---------------- Step 5: Final Result ----------------
print("\n✅ Local Minima occurs at x =", round(x, 4))
print("Function value at minima y =", round(function(x), 4))

# ---------------- Step 6: Visualization ----------------
import numpy as np
X = np.linspace(-10, 5, 100)
Y = function(X)

plt.figure(figsize=(8,5))
plt.plot(X, Y, label='y = (x + 3)^2', color='blue')
plt.scatter(x_values, [function(i) for i in x_values], color='red', label='Gradient Descent Steps')
plt.title('Gradient Descent to Find Local Minima')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
