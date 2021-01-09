import numpy as np
import matplotlib.pyplot as plt

# pick a value for the lens axis ~[0,1)
axis_ratio = 0.9999


def keeton(x, y, q):

    """
    Keeton's expression for SIE deflection angles, ignoring constants out the front

    x - grid of x positions
    y - grid of y positions
    q - axis ratio ~[0,1)
    """

    x_component = np.arctan((np.sqrt(1 - q**2)*x)/(np.sqrt((x**2)*(q**2) + y**2)))
    y_component = np.arctanh((np.sqrt(1 - q**2)*y)/(np.sqrt((x**2)*(q**2) + y**2)))

    return x_component, y_component


def kormann(x, y, q):

    """
    Kormann's expression for SIE deflection angles, ignoring constants out the front

    x - grid of x positions
    y - grid of y positions
    q - axis ratio ~[0,1)
    """

    x_component = np.arcsinh((np.sqrt(1 - q**2)*x)/(q*np.sqrt((q**2)*(x**2) + y**2)))
    y_component = np.arcsin((np.sqrt(1 - q**2)*y)/(np.sqrt((x**2)*(q**2) + y**2)))

    return x_component, y_component


xaxis = np.linspace(-5, 5, 1000)
yaxis = np.linspace(-5, 5, 1000)

print(xaxis)

alpha_keeton = keeton(x=xaxis[:,None], y=yaxis[None,:], q=axis_ratio)
alpha_kormann = kormann(x=xaxis[:,None], y=yaxis[None,:], q=axis_ratio)


fig = plt.figure()
plt.suptitle("q = {}".format(axis_ratio))
ax1 = fig.add_subplot(3,2,1)
ax1.set_title("Keeton (x component)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
im1 = ax1.imshow(alpha_keeton[0])
plt.colorbar(im1, ax=ax1)

ax2 = fig.add_subplot(3,2,3)
ax2.set_title("Kormann (x component)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
im2 = ax2.imshow(alpha_kormann[0])
plt.colorbar(im2, ax=ax2)

ax3 = fig.add_subplot(3,2,2)
ax3.set_title("Keeton (y component)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
im3 = ax3.imshow(alpha_keeton[1])
plt.colorbar(im3, ax=ax3)

ax4 = fig.add_subplot(3,2,4)
ax4.set_title("Kormann (y component)")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
im4 = ax4.imshow(alpha_kormann[1])
plt.colorbar(im4, ax=ax4)

ax5 = fig.add_subplot(3,2,5)
ax5.set_title("Keeton / Kormann (x component)")
ax5.set_xlabel("x")
ax5.set_ylabel("y")
im5 = ax5.imshow(alpha_keeton[0] / alpha_kormann[0])
plt.colorbar(im5, ax=ax5)

ax6 = fig.add_subplot(3,2,6)
ax6.set_title("Keeton / Kormann (y component)")
ax6.set_xlabel("x")
ax6.set_ylabel("y")
im6 = ax6.imshow(alpha_keeton[1] / alpha_kormann[1])
plt.colorbar(im6, ax=ax6)

plt.tight_layout()

plt.show()
plt.close()
