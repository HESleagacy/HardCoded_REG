# Classical Machine Learning — From Scratch & with Scikit-Learn

A structured exploration of fundamental machine learning algorithms, implemented both:

- 🔢 From scratch using NumPy
- 🧠 Using Scikit-Learn
- 📊 With visualizations via Matplotlib

This project walks through regression, optimization techniques, regularization, and classification using the Iris dataset.

---

## 📚 Topics Covered

### Regression
- Linear Regression (Normal Equation)
- Moore–Penrose Pseudoinverse
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Polynomial Regression
- Learning Curves
- Ridge Regression (L2)
- Lasso Regression (L1)
- Elastic Net

### Classification
- Logistic Regression (Binary)
- Softmax Regression (Multiclass)

---

## 🧮 Mathematical Foundations

The project demonstrates:

- Closed-form linear regression:
  
  θ = (XᵀX)⁻¹Xᵀy

- Gradient-based optimization:
  
  θ := θ − η∇J(θ)

- Regularized regression:
  
  Ridge: (XᵀX + αI)⁻¹Xᵀy

- Logistic function (sigmoid)
- Softmax function for multiclass classification
- Bias–variance tradeoff
- Underfitting vs Overfitting analysis

---

## 📈 1. Linear Regression

Synthetic dataset generated using:

y = 4 + 3x + Gaussian noise

Implemented via:

- Normal Equation
- Pseudoinverse
- Batch Gradient Descent
- Stochastic Gradient Descent

All approaches converge to similar parameter estimates.

---

## 📉 2. Polynomial Regression

Nonlinear data:

y = 0.45x² + x + 2 + noise

Feature transformation using:

```python
PolynomialFeatures(degree=2)
