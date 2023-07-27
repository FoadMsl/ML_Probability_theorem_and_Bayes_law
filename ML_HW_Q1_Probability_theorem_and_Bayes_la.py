#=================================================
#   ML_HW_Q1_Probability_theorem_and_Bayes_la
#   Bayes Theorem
#   Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
#   Using Python 3.9.16
#=================================================

# %clear
# %reset -f

#%%============================
""" Import Libraries """
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

#%%============================
import numpy as np
# Graduated Mean & Standard Deviation
mu_G = np.array([17, 16]) # Mean of G (Graduated)
Sigma_G = np.array([[2,1], [1,3]]) # Standard Deviation of G (Graduated)
# Not Graduated Mean & Standard Deviation
mu_nG = np.array([13, 12]) # Mean of nG (not Graduated)
Sigma_nG = np.array([[4,1], [1,5]]) # Standard Deviation of nG (not Graduated)
# Student Mean & Standard Deviation
mu_Sn = np.array([19, 18]) # Mean of Sn (Students)
Sigma_Sn = np.array([[1, 0.5], [0.5, 2]]) # Standard Deviation of Sn (Students)
# Priors
prior_G = 0.8
prior_nG = 0.1
prior_Sn = 0.1

#%%============================
""" Functions """
def gaussian(x, mu, Sigma):
    return 1/(2*np.pi*np.prod(Sigma))*np.exp(-0.5*np.dot((x-mu), np.dot(np.diag(1/Sigma**2), (x-mu).T)))

def classify(x, y):
    # Likelihoods
    likelihood_G = gaussian(np.array([x, y]), mu_G, Sigma_G)
    likelihood_nG = gaussian(np.array([x, y]), mu_nG, Sigma_nG)
    likelihood_Sn = gaussian(np.array([x, y]), mu_Sn, Sigma_Sn)
    # Posteriors
    posterior_G = likelihood_G * prior_G
    posterior_nG = likelihood_nG * prior_nG
    posterior_Sn = likelihood_Sn * prior_Sn
    # Probabilities
    probs=[posterior_G, posterior_nG, posterior_Sn]
    label_idx=np.argmax(probs)
    return label_idx

#%%============================
""" Plots """
import matplotlib.pyplot as plt

""" plot each class with its boundaries in a 2D space """
# Figure 1
x = np.linspace(0, 25, 250) # Set line space in x axis
y = np.linspace(0, 25, 250) # Set line space in y axis
X, Y = np.meshgrid(x, y) # Return coordinate matrices from coordinate vectors.
Z = np.zeros_like(X) # Return an array of zeros with the same shape and type as a given array.
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j]= classify(X[i,j], Y[i,j])

plt.contourf(X, Y, Z, alpha=0.5) # Draw contour lines and filled contours, respectively | alpha: The alpha blending value, between 0 (transparent) and 1 (opaque)
plt.scatter([17, 13, 19], [16, 12, 18], c=['red', 'blue', 'green'], edgecolors='face') # A scatter plot of y vs. x with varying marker size and/or color. | c: array-like or list of colors or color. | edgecolors: The edge color of the marker.. 
plt.title("Gaussian Bayes Classifier")
plt.xlabel("ML Exam Score")
plt.ylabel("High School Class Score")
plt.legend(["Graduated", "Not Graduated", "Student"])
plt.show()

""" plot Gaussian probabilities for each class in a 3D space """
# Figure 2
from scipy import stats # Statistical functions
# This module contains a large number of probability distributions, summary and 
# frequency statistics, correlation functions and statistical tests, masked statistics, 
# kernel density estimation, quasi-Monte Carlo functionality, and more.

# Graduated
x_G, y_G = np.meshgrid(np.linspace(mu_G[0]-3*np.sqrt(Sigma_G[0,0]), mu_G[0]+3*np.sqrt(Sigma_G[0,0]), 100),
                       np.linspace(mu_G[1]-3*np.sqrt(Sigma_G[1,1]), mu_G[1]+3*np.sqrt(Sigma_G[1,1]), 100))
z_G = np.reshape(stats.multivariate_normal(mu_G, Sigma_G).pdf(np.column_stack([x_G.ravel(), y_G.ravel()])), x_G.shape)
# Not Graduated
x_nG, y_nG = np.meshgrid(np.linspace(mu_nG[0]-3*np.sqrt(Sigma_nG[0,0]), mu_nG[0]+3*np.sqrt(Sigma_nG[0,0]), 100),
                         np.linspace(mu_nG[1]-3*np.sqrt(Sigma_nG[1,1]), mu_nG[1]+3*np.sqrt(Sigma_nG[1,1]), 100))
z_nG = np.reshape(stats.multivariate_normal(mu_nG, Sigma_nG).pdf(np.column_stack([x_nG.ravel(), y_nG.ravel()])), x_nG.shape)
# Student
x_Sn, y_Sn = np.meshgrid(np.linspace(mu_Sn[0]-3*np.sqrt(Sigma_Sn[0,0]), mu_Sn[0]+3*np.sqrt(Sigma_Sn[0,0]), 100),
                         np.linspace(mu_Sn[1]-3*np.sqrt(Sigma_Sn[1,1]), mu_Sn[1]+3*np.sqrt(Sigma_Sn[1,1]), 100))
z_Sn = np.reshape(stats.multivariate_normal(mu_Sn, Sigma_Sn).pdf(np.column_stack([x_Sn.ravel(), y_Sn.ravel()])), x_Sn.shape)

Figure2 = plt.figure()
ax = Figure2.add_subplot(121, projection='3d')
ax.plot_surface(x_G, y_G, z_G, cmap='Reds', alpha=0.5)
ax.plot_surface(x_nG, y_nG, z_nG, cmap='Blues', alpha=0.5)
ax.plot_surface(x_Sn, y_Sn, z_Sn, cmap='Greens', alpha=0.5)
ax.set_xlabel('ML Score')
ax.set_ylabel('High School Class Score')
ax.set_zlabel('Probability Density')
ax.set_title('Gaussian Probabilities For Each Class')
plt.show()