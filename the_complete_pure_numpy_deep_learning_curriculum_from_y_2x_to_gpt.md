# The Complete Pure NumPy Deep Learning Curriculum: From $y = 2x$ to GPT

## Master Goal

Your mission is to build a fully working, autoregressive **Generative Pre-trained Transformer (GPT)** from scratch using **only Python and NumPy**. Zero deep learning libraries (`PyTorch`, `TensorFlow`, `JAX`, `scikit-learn`) are allowed.

Every project below is a **100% standalone, self-contained experiment**. Instead of building a giant software library or importing files from earlier lessons, each project is its own independent script where you implement the math directly for that specific problem, train it on its own dataset, and verify it against concrete test values.

Each project follows the same three-part structure:

1. **The New Concept:** The exact single mathematical or architectural idea introduced in this script.

2. **Deep-Dive Theory:** Full mathematical derivations, intuition, and matrix dimensions.

3. **Project Specification:** What your standalone script must accomplish, the **Training Data**, and the **Test Values** to prove your implementation works.

## Phase 1: The Absolute Basics (Scalars, Calculus & Linear Models)

### Project 1: The Simplest Possible Neuron ($y = 2x$, No Bias)

* **New Concept:** Forward pass with a single parameter $w$, Squared Error loss, the derivative of a parabola, and stepping down a slope (Gradient Descent).

* **Deep-Dive Theory:**
  Imagine a machine with a single dial $w$. You give it a number $x$, and it multiplies $x$ by the dial setting $w$ to make a guess $\hat{y}$:

  $$
  \hat{y} = w \cdot x
  $$

  Suppose the true answer is $y = 2x$, so the dial *should* be set to $w = 2$, but right now it is set randomly to $w = 0.5$. How do we measure how bad the guess is? We compute the **Mean Squared Error (MSE)** across all $N$ training examples:

  $$
  \mathcal{L}(w) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (w x_i - y_i)^2
  $$

  Why square the error? Because errors can be positive or negative, squaring makes all errors positive, punishes big mistakes much harder than small ones, and creates a smooth U-shaped curve (a parabola) with a single lowest point.

  To find which way is "downhill" on that parabola, we take the derivative of $\mathcal{L}$ with respect to $w$ using the **Chain Rule** ($\frac{d}{dw}[f(g(w))] = f'(g(w)) \cdot g'(w)$):

  $$
  \frac{d\mathcal{L}}{dw} = \frac{1}{N} \sum_{i=1}^{N} 2(w x_i - y_i) \cdot x_i = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_i
  $$

  * **If** $\frac{d\mathcal{L}}{dw} > 0$**:** Increasing $w$ increases error, so we must *decrease* $w$.

  * **If** $\frac{d\mathcal{L}}{dw} < 0$**:** Increasing $w$ decreases error, so we must *increase* $w$.

  We update $w$ using a small step size called the **Learning Rate** ($\eta$, e.g., $0.01$):

  $$
  w \leftarrow w - \eta \frac{d\mathcal{L}}{dw}
  $$

* **Project Specification:**

  * **What the script should do:** Start with $w = 0.0$ and learn the rule $y = 2x$ purely via gradient descent.

  * **Training Data:**

    $$
    X_{\text{train}} = [1.0, 2.0, 3.0, 4.0], \quad Y_{\text{train}} = [2.0, 4.0, 6.0, 8.0]
    $$

  * **Test Values:** Input $X_{\text{test}} = [5.0, -3.0, 10.0]$. Expected output: $\hat{Y} \approx [10.0, -6.0, 20.0]$ (learned $w \approx 2.000$).

### Project 2: Adding a Bias Term ($y = 2x + 1$)

* **New Concept:** Partial derivatives ($\partial$), multiple parameters ($w$ and $b$), and shifting a line off the origin.

* **Deep-Dive Theory:**
  Without a bias term, $\hat{y} = wx$ is forced to always pass through $(0, 0)$. To fit any straight line, we add a **bias** $b$:

  $$
  \hat{y} = w \cdot x + b
  $$

  Now our loss function $\mathcal{L}(w, b)$ is a 3D bowl depending on *two* variables. We take **partial derivatives**—treating $b$ as a constant when differentiating with respect to $w$, and treating $w$ as a constant when differentiating with respect to $b$:

  $$
  \frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i
  $$

  $$
  \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot 1
  $$

  Both parameters update simultaneously at every step:

  $$
  w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
  $$

* **Project Specification:**

  * **What the script should do:** Learn both slope and intercept for $y = 2x + 1$ starting from $w = 0, b = 0$.

  * **Training Data:**

    $$
    X_{\text{train}} = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    $$

    $$
    Y_{\text{train}} = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
    $$

  * **Test Values:** Input $X_{\text{test}} = [4.0, -5.0, 0.5]$. Expected output: $\hat{Y} \approx [9.0, -9.0, 2.0]$.

### Project 3: Multiple Inputs & Vectorization (Matrix Multiplication)

* **New Concept:** Representing datasets as matrices, dot products (`@`), broadcasting, and eliminating Python `for`-loops over samples.

* **Deep-Dive Theory:**
  Real models take many input features at once (e.g., $x_1, x_2, x_3$). For a single sample, $\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$.
  In linear algebra, we stack $N$ samples (each with $D$ features) into an input matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, and put our weights into a column vector $\mathbf{W} \in \mathbb{R}^{D \times 1}$:

  $$
  \hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + b
  $$

  NumPy automatically **broadcasts** the scalar $b$ across all $N$ rows.
  By the rules of matrix calculus, summing over all $N$ samples happens automatically when we multiply $\mathbf{X}^T$ (shape $D \times N$) by the error vector $(\hat{\mathbf{Y}} - \mathbf{Y})$ (shape $N \times 1$):

  $$
  \nabla_{\mathbf{W}} \mathcal{L} = \frac{2}{N} \mathbf{X}^T (\hat{\mathbf{Y}} - \mathbf{Y}) \quad (\text{shape } D \times 1)
  $$

  $$
  \nabla_{b} \mathcal{L} = \frac{2}{N} \sum_{i=1}^{N} (\hat{\mathbf{Y}}_i - \mathbf{Y}_i) \quad (\text{scalar})
  $$

* **Project Specification:**

  * **What the script should do:** Learn the 3-variable plane $y = 3x_1 - 2x_2 + 0.5x_3 + 4$ using purely vectorized matrix operations (`np.dot` or `@`).

  * **Training Data:**

    $$
    \mathbf{X}_{\text{train}} = \begin{bmatrix} 1 & 1 & 2 \\ 2 & 0 & -2 \\ -1 & 3 & 4 \\ 0 & -2 & 2 \\ 3 & 1 & 0 \\ -2 & -1 & -4 \end{bmatrix}, \quad \mathbf{Y}_{\text{train}} = \begin{bmatrix} 6.0 \\ 9.0 \\ -3.0 \\ 9.0 \\ 11.0 \\ -2.0 \end{bmatrix}
    $$

  * **Test Values:**

    $$
    \mathbf{X}_{\text{test}} = \begin{bmatrix} 1 & 2 & 4 \\ -1 & -1 & 2 \end{bmatrix}
    $$

    Expected output: $\hat{\mathbf{Y}} \approx \begin{bmatrix} 5.0 \\ 4.0 \end{bmatrix}$.

## Phase 2: Probability, Activations & Classification

### Project 4: Binary Classification (The Sigmoid Neuron & Log Loss)

* **New Concept:** Squashing numbers into probabilities $(0, 1)$ with Sigmoid, and Binary Cross-Entropy (Log Loss).

* **Deep-Dive Theory:**
  What if we want to answer a Yes/No question (Class `1` vs Class `0`)? A linear equation $\mathbf{X}\mathbf{W} + b$ can output $+500$ or $-200$, which aren't valid probabilities. We wrap the linear output $z = \mathbf{X}\mathbf{W} + b$ inside the **Sigmoid** function:

  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$

  * When $z \to +\infty$, $\sigma(z) \to 1$.

  * When $z = 0$, $\sigma(z) = 0.5$.

  * When $z \to -\infty$, $\sigma(z) \to 0$.

  Its derivative is remarkably clean:

  $$
  \sigma'(z) = \sigma(z)(1 - \sigma(z))
  $$

  If we used MSE loss with Sigmoid, the loss surface would become wavy and flat when the neuron is confident and wrong (vanishing gradient). Instead, we use **Binary Cross-Entropy (BCE)**:

  $$
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i + \epsilon) + (1 - y_i) \log(1 - \hat{y}_i + \epsilon) \right]
  $$

  *(We add* $\epsilon = 10^{-15}$ *inside* $\log$ *so Python never computes* $\log(0) = -\infty$*).*
  When you differentiate BCE with respect to $z$ ($\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \sigma'(z)$), the $\sigma(z)(1-\sigma(z))$ terms cancel out completely, leaving:

  $$
  \frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y
  $$

  Therefore, the weight and bias gradients remain identical in form to linear regression:

  $$
  \nabla_{\mathbf{W}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{Y}} - \mathbf{Y}), \quad \nabla_{b} \mathcal{L} = \frac{1}{N} \sum (\hat{\mathbf{Y}} - \mathbf{Y})
  $$

* **Project Specification:**

  * **What the script should do:** Classify 2D points as Class `1` if $x_1 + x_2 > 0$, and Class `0` otherwise.

  * **Training Data:**

    $$
    \mathbf{X}_{\text{train}} = \begin{bmatrix} 2.0 & 3.0 \\ 1.0 & 1.5 \\ 0.5 & 0.2 \\ -2.0 & -3.0 \\ -1.0 & -0.5 \\ -0.2 & -0.8 \\ 1.5 & -0.5 \\ -1.5 & 0.5 \end{bmatrix}, \quad \mathbf{Y}_{\text{train}} = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 0 \\ 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}
    $$

  * **Test Values:**

    $$
    \mathbf{X}_{\text{test}} = \begin{bmatrix} 3.0 & 1.0 \\ -2.5 & -1.0 \\ 0.4 & -0.1 \end{bmatrix}
    $$

    Expected output: Probabilities $> 0.9$ for row 1, $< 0.1$ for row 2, and $> 0.5$ for row 3.

### Project 5: Multi-Class Classification (Softmax & Categorical Cross-Entropy)

* **New Concept:** Predicting among $C$ choices at once, Softmax, the Log-Sum-Exp numerical stability trick, and One-Hot Encoding.

* **Deep-Dive Theory:**
  A GPT model has to choose the next word out of $V = 50,000$ possible words. That is a multi-class classification problem!
  Given $C$ classes, our linear layer outputs $C$ raw scores (called **logits**) per sample: $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b} \in \mathbb{R}^{N \times C}$.
  We convert each row of logits $\mathbf{z} = [z_1, \dots, z_C]$ into a probability distribution that sums to $1.0$ using **Softmax**:

  $$
  \text{Softmax}(z_c) = \frac{e^{z_c - \max(\mathbf{z})}}{\sum_{j=1}^{C} e^{z_j - \max(\mathbf{z})}}
  $$

  * **Why subtract** $\max(\mathbf{z})$**?** If a logit is $1000$, `np.exp(1000)` overflows 64-bit floats and returns `inf`, resulting in `inf / inf = NaN`. Subtracting $\max(\mathbf{z})$ makes the largest exponent $e^0 = 1$, guaranteeing numerical safety without changing the output probabilities at all!

  We represent true integer class labels (e.g., `2`) as **One-Hot vectors** (e.g., `[0, 0, 1]`) in matrix $\mathbf{Y} \in \{0, 1\}^{N \times C}$. The **Categorical Cross-Entropy** loss is:

  $$
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} Y_{i,c} \log(\hat{Y}_{i,c} + \epsilon)
  $$

  Once again, the derivative with respect to the raw logits $\mathbf{Z}$ simplifies to:

  $$
  \nabla_{\mathbf{Z}} \mathcal{L} = \hat{\mathbf{Y}} - \mathbf{Y}
  $$

* **Project Specification:**

  * **What the script should do:** Classify 2D points into 3 quadrant clusters (Class `0`, `1`, or `2`).

  * **Training Data:**

    $$
    \mathbf{X}_{\text{train}} = \begin{bmatrix} 2.0 & 2.0 \\ 1.5 & 3.0 \\ 3.0 & 1.0 \\ -2.0 & 2.0 \\ -3.0 & 1.5 \\ -1.5 & 2.5 \\ -2.0 & -2.0 \\ -1.0 & -3.0 \\ -3.0 & -1.0 \end{bmatrix}, \quad \mathbf{Y}_{\text{labels}} = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    $$

  * **Test Values:**

    $$
    \mathbf{X}_{\text{test}} = \begin{bmatrix} 2.5 & 2.5 \\ -2.5 & 2.0 \\ -2.5 & -2.5 \end{bmatrix}
    $$

    Expected output: Predicted classes (`np.argmax`) `[0, 1, 2]` with $> 0.95$ probability on the true class.

## Phase 3: Deep Learning Mechanics & Optimization

### Project 6: Breaking Linearity — The XOR Gate (2-Layer MLP & Backprop)

* **New Concept:** Why stacking linear layers without non-linearities fails, ReLU activation, He Weight Initialization, and Backpropagation.

* **Deep-Dive Theory:**
  If you stack two linear layers without an activation function between them:

  $$
  \mathbf{Y} = (\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 = \mathbf{X}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2) = \mathbf{X}\mathbf{W}_{\text{combined}} + \mathbf{b}_{\text{combined}}
  $$

  Two linear layers collapse into a *single* linear layer! It still cannot bend space to solve non-linear problems like **XOR**.
  To bend space, we insert a non-linear activation function—**ReLU (Rectified Linear Unit)**—after the hidden layer:

  $$
  \text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
  $$

  **Weight Initialization:** Never initialize weights to all zeros in a multi-layer network! If every neuron starts at `0`, every neuron computes the exact same gradient and updates identically forever (**Symmetry Problem**). Instead, use **He Initialization**:

  $$
  \mathbf{W} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)
  $$

  **Forward Pass:**

  $$
  \mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1 \quad \to \quad \mathbf{A}_1 = \text{ReLU}(\mathbf{Z}_1) \quad \to \quad \mathbf{Z}_2 = \mathbf{A}_1\mathbf{W}_2 + \mathbf{b}_2 \quad \to \quad \hat{\mathbf{Y}} = \sigma(\mathbf{Z}_2)
  $$

  **Backward Pass (Backpropagation):**
  We send the error signal backward through the chain rule:

  $$
  \delta_2 = \hat{\mathbf{Y}} - \mathbf{Y}
  $$

  $$
  \nabla_{\mathbf{W}_2}\mathcal{L} = \frac{1}{N} \mathbf{A}_1^T \delta_2, \quad \nabla_{\mathbf{b}_2}\mathcal{L} = \frac{1}{N} \sum \delta_2
  $$

  $$
  \delta_1 = (\delta_2 \mathbf{W}_2^T) \odot \text{ReLU}'(\mathbf{Z}_1)
  $$

  $$
  \nabla_{\mathbf{W}_1}\mathcal{L} = \frac{1}{N} \mathbf{X}^T \delta_1, \quad \nabla_{\mathbf{b}_1}\mathcal{L} = \frac{1}{N} \sum \delta_1
  $$

  *(Here* $\odot$ *means element-wise multiplication).*

* **Project Specification:**

  * **What the script should do:** Solve the XOR truth table using a standalone 2-layer MLP ($2 \to 8 \to 1$).

  * **Training Data:**

    $$
    \mathbf{X}_{\text{train}} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{Y}_{\text{train}} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix}
    $$

  * **Test Values:** Predict on $\mathbf{X}_{\text{train}}$. Expected output: $\hat{\mathbf{Y}} \approx \begin{bmatrix} 0.01 \\ 0.99 \\ 0.99 \\ 0.01 \end{bmatrix}$.

### Project 7: Numerical Gradient Checking (Finite Differences vs. Calculus)

* **New Concept:** Verifying analytical backpropagation formulas using the limit definition of a derivative.

* **Deep-Dive Theory:**
  As networks get deeper, deriving gradients by hand becomes prone to subtle bugs (a missing transpose, a wrong axis in `np.sum`, or a forgotten factor of $2$). You can verify any calculus formula using **Numerical Gradient Checking**.

  Recall the central-difference definition of a derivative:

  $$
  \frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \delta) - \mathcal{L}(\theta_i - \delta)}{2\delta}
  $$

  where $\delta = 10^{-5}$. By nudging a single weight up by $\delta$, computing the forward loss, nudging it down by $\delta$, computing the forward loss again, and taking the difference divided by $2\delta$, we get the *ground truth* slope without using any calculus formulas!

  We compare the analytical gradient vector $\mathbf{g}_{\text{ana}}$ against the numerical gradient vector $\mathbf{g}_{\text{num}}$ using **Relative Error**:

  $$
  \text{Relative Error} = \frac{\|\mathbf{g}_{\text{ana}} - \mathbf{g}_{\text{num}}\|_2}{\|\mathbf{g}_{\text{ana}}\|_2 + \|\mathbf{g}_{\text{num}}\|_2}
  $$

  If $\text{Relative Error} < 10^{-7}$, the analytical calculus is exact.

* **Project Specification:**

  * **What the script should do:** Write a standalone script that creates a 2-layer network ($\mathbf{X} \to \text{Linear} \to \tanh \to \text{Linear} \to \text{MSE}$), computes analytical gradients via backprop, computes numerical gradients via finite differences, and prints the relative error for every weight matrix and bias vector.

  * **Training Data:** A single random input batch $\mathbf{X} \in \mathbb{R}^{3 \times 4}$ and random target $\mathbf{Y} \in \mathbb{R}^{3 \times 2}$.

  * **Test Values:** Confirm that `Relative Error < 1e-7` for $\mathbf{W}_1$, $\mathbf{b}_1$, $\mathbf{W}_2$, $\mathbf{b}_2$, and $\mathbf{X}$.

### Project 8: Deep Curve Fitting & Mini-Batch Stochastic Gradient Descent

* **New Concept:** Training a 3-layer network ($1 \to 32 \to 32 \to 1$) on a continuous non-linear curve, dataset shuffling, and Mini-Batch SGD.

* **Deep-Dive Theory:**
  In earlier projects, we updated weights once per full pass over 4–9 data points (**Batch Gradient Descent**). When datasets have hundreds or thousands of points, we shuffle the dataset every epoch and slice it into small chunks of size $B$ (e.g., `batch_size = 32`):

  $$
  \theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}_{\text{batch}}
  $$

  Mini-batches give much faster updates per epoch and introduce helpful gradient noise that prevents the network from getting stuck in poor local minima.

* **Project Specification:**

  * **What the script should do:** Build a self-contained 3-layer MLP ($1 \to 32 \to 32 \to 1$ with `ReLU` activations and MSE loss) and train it using shuffled mini-batches of size `32` to approximate the quadratic curve $y = x^2 - 2x + 1$.

  * **Training Data:** 256 evenly spaced points $x \in [-3, 3]$, $y = x^2 - 2x + 1$.

  * **Test Values:** $X_{\text{test}} = [-2.0, 0.0, 1.0, 3.0]$. Expected output: $\hat{Y} \approx [9.0, 1.0, 0.0, 4.0]$ (Test MSE $< 0.01$).

### Project 9: The Adam Optimizer — Solving the 2D Spiral Classification Challenge

* **New Concept:** Momentum, RMSprop scale normalization, bias correction, and per-parameter adaptive learning rates on a twisted multi-class geometry.

* **Deep-Dive Theory:**
  Standard gradient descent uses one fixed learning rate $\eta$ for all weights, which struggles when loss landscapes have steep ravines in one direction and flat plateaus in another—such as when carving intertwined spiral decision boundaries in 2D space. **Adam** adapts the learning rate individually for every single parameter by tracking two running statistics:

  * $m_t$: The **1st moment** (exponential moving average of gradients $\to$ momentum).

  * $v_t$: The **2nd moment** (exponential moving average of squared gradients $\to$ scale normalization).

  At step $t = 1, 2, 3, \dots$ (with standard hyperparameters $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$):

  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
  $$

  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
  $$

  Because $m_0 = 0$ and $v_0 = 0$, early estimates are biased toward zero, so we apply **bias correction**:

  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$

  $$
  \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$

* **Project Specification:**

  * **What the script should do:** Generate a classic **3-Class 2D Spiral Dataset** (300 points, 100 per spiral arm twisting outward from the origin) and train a 3-layer classifier ($2 \to 64 \to 64 \to 3$ with `ReLU` + `Softmax` Cross-Entropy) using your own `Adam` update loop, comparing its convergence speed and final accuracy against plain SGD over 500 epochs.

  * **Training Data:** For class $c \in \{0, 1, 2\}$ and $i \in [0, 99]$: radius $r = i / 100$, angle $\theta = 4r + \frac{2\pi c}{3} + \text{noise}$, with coordinates $(x_1, x_2) = (r\sin\theta, r\cos\theta)$.

  * **Test Values:** Verify that Adam reaches $> 95\%$ training accuracy within 500 epochs (`lr = 0.01`), significantly outperforming plain SGD at the same learning rate.

## Phase 4: Language Foundations — Tokenization, Embeddings & Sequence Modeling

### Project 10: Subword Tokenization (Byte-Pair Encoding / BPE)

* **New Concept:** Turning raw human text into integer token IDs, vocabulary construction, and **Byte-Pair Encoding (BPE)**—how real GPT models read text.

* **Deep-Dive Theory:**
  Neural networks only understand numbers. How do we convert `"Hello world!"` into a list of integers?

  * **Character-Level:** Assign every unique character an ID. Problem: Sequences become very long and the model must learn spelling from scratch.

  * **Word-Level:** Assign every word an ID. Problem: Vocabulary explodes to millions of words, and variations like `"run"`, `"running"`, `"runs"` share no structure.

  * **Subword / BPE (Used by GPT):** Finds the sweet spot! Common words like `" the"` get a single token ID, while rare words split into meaningful pieces like `"un"`, `"break"`, `"able"`.

  **How BPE Training Works:**

  1. Start with a base vocabulary of all unique individual characters in the training text (size $V_0$).

  2. Convert the training text into a list of these base integer IDs.

  3. Count every adjacent pair of token IDs $(id_1, id_2)$ across the sequence.

  4. Pick the most frequent pair $(id_1, id_2)$, mint a new token ID $id_{\text{new}} = V_{\text{current}}$, and record the merge rule $(id_1, id_2) \to id_{\text{new}}$.

  5. Replace every occurrence of $(id_1, id_2)$ in the sequence with $id_{\text{new}}$.

  6. Repeat steps 3–5 until reaching your target `vocab_size`.

  **How BPE Encoding Works on New Text:**
  Convert characters to base IDs, then apply your saved merge rules *in the exact order they were learned* during training!

* **Project Specification:**

  * **What the script should do:** Implement a standalone BPE tokenizer (`train`, `encode`, and `decode` functions) in a single script and inspect the learned subword vocabulary.

  * **Training Data:**
    `"low low low low low lower lower newest newest newest newest newest newest widest widest widest"`

  * **Test Values:**
    Train with `vocab_size = 18` (11 unique characters + 7 merges).

    1. Verify that `encode("lowest")` produces fewer than 6 tokens (confirming `"low"` and `"est"` were merged).

    2. Verify that `decode(encode("lowest newest widest")) == "lowest newest widest"` (lossless round-trip).

### Project 11: Word Embeddings & Word Similarity (Skip-Gram Word2Vec)

* **New Concept:** Dense vector representations of words, embedding matrix lookup, sparse gradient accumulation (`np.add.at`), and cosine similarity.

* **Deep-Dive Theory:**
  Suppose `"cat"` is token ID `5` and `"dog"` is token ID `900`. We must **never** feed raw token IDs into a linear layer—that would imply `"dog"` is $180\times$ larger than `"cat"`!

  Instead, we create a learnable **Embedding Table** $\mathbf{E} \in \mathbb{R}^{V \times d}$, where each row $i$ is a learnable vector representing word $i$.

  * **Forward Lookup:** Given a batch of integer word IDs $\mathbf{x} \in \mathbb{Z}^{B}$, look up rows: $\mathbf{H} = \mathbf{E}[\mathbf{x}] \in \mathbb{R}^{B \times d}$.

  * **Backward Lookup:** Given upstream gradient $\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}}$, only the looked-up rows receive gradients.
    *Critical NumPy Detail:* If word ID `5` appears multiple times in $\mathbf{x}$, standard indexing `dE[x] += G` overwrites duplicates instead of summing them! You **must** use `np.add.at(dE, x, G)`.

  In **Skip-Gram Word2Vec**, a simple network ($\mathbf{E}[\mathbf{x}] \to \mathbf{W}_{\text{out}} \to \text{Softmax}$) learns to predict neighboring context words from a center word. Words that appear in similar contexts naturally cluster together in embedding space!

* **Project Specification:**

  * **What the script should do:** Train a standalone Skip-Gram Word2Vec model ($d = 8$) on a small synthetic corpus of related sentences, and compute **Cosine Similarity** ($\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$) between learned word vectors.

  * **Training Data:**
    Repeat 100 times: `"the king rules the kingdom . the queen rules the kingdom . the dog barks loud . the cat meows loud ."`

  * **Test Values:** Verify that `cosine_similarity(vec("king"), vec("queen")) > cosine_similarity(vec("king"), vec("dog"))` and `cosine_similarity(vec("dog"), vec("cat")) > cosine_similarity(vec("dog"), vec("kingdom"))`.

### Project 12: Neural N-Gram Language Model (Your First Text Generator!)

* **New Concept:** Sliding-window sequence datasets, concatenating multiple token embeddings, and autoregressive next-token generation.

* **Deep-Dive Theory:**
  Before building Transformers, we build Bengio's classic **Neural Probabilistic Language Model**. Given a fixed context window of $C = 3$ previous characters $[x_{t-3}, x_{t-2}, x_{t-1}]$, the network predicts the probability distribution of the next character $x_t$:

  1. Look up embeddings for the $C$ characters: shape $(B, C, d)$.

  2. Flatten/concatenate them into a single vector per sample: shape $(B, C \cdot d)$.

  3. Pass through `Linear` $\to$ `ReLU` $\to$ `Linear` $\to$ `Softmax` to output logits over the character vocabulary $V$.

  4. Train with Categorical Cross-Entropy loss and Adam.

* **Project Specification:**

  * **What the script should do:** Build a self-contained character-level Neural Language Model ($C = 3$) and generate text autoregressively by repeatedly predicting the next character and sliding the 3-character window forward.

  * **Training Data:**
    `"hello world. hello deep learning. hello numpy. hello gpt. "` (repeated 50 times).

  * **Test Values:**
    Feed prompt `"hel"` and autoregressively append the predicted character (`argmax`) 9 times. Expected output: `"hello worl"` or `"hello nump"`.

## Phase 5: Building the Transformer Components

### Project 13: Layer Normalization & Residual Connections

* **New Concept:** Stabilizing deep activations across features (`LayerNorm`), analytical LayerNorm derivatives, and gradient highways (`Residual Connections`).

* **Deep-Dive Theory:**
  As signals pass through deep networks, numbers can explode or vanish. **Layer Normalization** normalizes each token's feature vector across its $d$ features to mean $0$ and variance $1$, then applies learnable scale $\gamma \in \mathbb{R}^{d}$ and shift $\beta \in \mathbb{R}^{d}$:

  $$
  \mu = \frac{1}{d} \sum_{j=1}^{d} x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2
  $$

  $$
  \hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_j = \gamma_j \hat{x}_j + \beta_j
  $$

  **Analytical Backward Pass for LayerNorm:**
  Given upstream gradient $\mathbf{g} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ and $\mathbf{g}_{\hat{x}} = \mathbf{g} \odot \gamma$:

  $$
  \nabla_{\gamma}\mathcal{L} = \sum_{\text{batch, seq}} \mathbf{g} \odot \hat{\mathbf{x}}, \quad \nabla_{\beta}\mathcal{L} = \sum_{\text{batch, seq}} \mathbf{g}
  $$

  $$
  \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{1}{d \sqrt{\sigma^2 + \epsilon}} \left( d \cdot \mathbf{g}_{\hat{x}} - \sum_{j} g_{\hat{x}, j} - \hat{\mathbf{x}} \odot \sum_{j} (g_{\hat{x}, j} \cdot \hat{x}_j) \right)
  $$

  **Residual Connections:**
  Wrapping sub-layers as $\mathbf{x}_{\text{out}} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$ splits the backward gradient into two paths: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} + \frac{\partial \mathcal{L}}{\partial \text{Sublayer}}\frac{\partial \text{Sublayer}}{\partial \mathbf{x}}$, preventing vanishing gradients in very deep networks.

* **Project Specification:**

  * **What the script should do:** Write a standalone script implementing `LayerNorm` forward and backward passes, verify `dx`, `dgamma`, and `dbeta` against finite differences, and train a deep 10-layer residual MLP ($\mathbf{x} \leftarrow \mathbf{x} + \text{Linear}(\text{ReLU}(\text{LayerNorm}(\mathbf{x})))$) on a simple regression task to see clean gradient flow across 10 layers.

  * **Training Data:** Random tensor $\mathbf{X} \in \mathbb{R}^{4 \times 6 \times 16}$ for finite-difference gradient verification.

  * **Test Values:** Confirm `Relative Error < 1e-7` for `dx`, `dgamma`, and `dbeta`.

### Project 14: Scaled Dot-Product Attention & Causal Masking

* **New Concept:** Queries ($\mathbf{Q}$), Keys ($\mathbf{K}$), Values ($\mathbf{V}$), attention score scaling, causal upper-triangular masking, and backpropagating through batched Softmax.

* **Deep-Dive Theory:**
  In Project 12, the network could only look at a rigid window of 3 tokens. **Self-Attention** lets every token dynamically search all previous tokens in the sequence and pull relevant information forward!

  Given matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times T \times d_k}$:

  1. **Similarity Scores:**

     $$
     \mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times T \times T}
     $$

     *(Dividing by* $\sqrt{d_k}$ *keeps variance near* $1$ *so Softmax doesn't saturate).*

  2. **Causal Masking:** Token at position $t$ may only attend to positions $\le t$. Add an upper-triangular mask $\mathbf{M}$ where $M_{i,j} = 0$ if $i \ge j$ and $M_{i,j} = -10^9$ if $i < j$:

     $$
     \mathbf{S}_{\text{masked}} = \mathbf{S} + \mathbf{M}
     $$

  3. **Attention Weights & Mixing:**

     $$
     \mathbf{A} = \text{Softmax}(\mathbf{S}_{\text{masked}}), \quad \mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{B \times T \times d_k}
     $$

  **Backward Pass:**
  Given $\mathbf{G}_O = \frac{\partial \mathcal{L}}{\partial \mathbf{O}}$:

  1. $\mathbf{G}_V = \mathbf{A}^T \mathbf{G}_O$

  2. $\mathbf{G}_A = \mathbf{G}_O \mathbf{V}^T$

  3. **Softmax Backward:** $\mathbf{G}_S = \mathbf{A} \odot \left( \mathbf{G}_A - \sum_{\text{last axis}} (\mathbf{G}_A \odot \mathbf{A}) \right)$

  4. $\mathbf{G}_Q = \frac{\mathbf{G}_S \mathbf{K}}{\sqrt{d_k}}, \quad \mathbf{G}_K = \frac{\mathbf{G}_S^T \mathbf{Q}}{\sqrt{d_k}}$

* **Project Specification:**

  * **What the script should do:** Write a standalone script implementing causal scaled dot-product attention (`forward` and `backward`), verify gradients with finite differences, and print the attention weight matrix $\mathbf{A}$ to confirm all entries above the main diagonal are `0.0`.

  * **Training Data:** Random tensors $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{2 \times 5 \times 8}$.

  * **Test Values:** Confirm `Relative Error < 1e-7` for $\mathbf{G}_Q, \mathbf{G}_K, \mathbf{G}_V$ and `np.allclose(np.triu(A, k=1), 0.0)`.

### Project 15: Single-Head Self-Attention & Associative Recall

* **New Concept:** Learnable $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ projections and training a self-attention model to dynamically retrieve information across arbitrary distances.

* **Deep-Dive Theory:**
  We connect learnable projection matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ to scaled dot-product attention:

  $$
  \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V
  $$

  * **Query (**$\mathbf{Q}$**):** *"What information am I looking for?"*

  * **Key (**$\mathbf{K}$**):** *"What information do I contain?"*

  * **Value (**$\mathbf{V}$**):** *"If we match, here is the content I pass forward."*

  During backprop, the input gradient sums contributions from all three paths:

  $$
  \mathbf{G}_X = \mathbf{G}_Q \mathbf{W}_Q^T + \mathbf{G}_K \mathbf{W}_K^T + \mathbf{G}_V \mathbf{W}_V^T
  $$

* **Project Specification:**

  * **What the script should do:** Build a self-contained single-head attention model (`Embedding` + `CausalSelfAttention` + `Linear`) and train it on **Associative Recall**: locate marker token `4` earlier in the sequence and output whichever token came right after it.

  * **Training Data:** 500 random sequences of length 6 from `{0, 1, 2, 3, 4}` where `4` appears once at index `0, 1, 2, or 3` and again at index `5`. Example: `[1, 4, 2, 0, 3, 4]` $\to$ target at index `5` is `2`.

  * **Test Values:**

    1. `[0, 4, 3, 1, 2, 4]` $\to$ Expected prediction at final step: `3`.

    2. `[2, 1, 4, 0, 3, 4]` $\to$ Expected prediction at final step: `0`.

### Project 16: Positional Embeddings & Multi-Head Attention (MHA)

* **New Concept:** Teaching the Transformer token order via Positional Embeddings, and running $H$ attention heads in parallel via 4D tensor reshaping.

* **Deep-Dive Theory:**

  1. **Positional Embeddings:** Self-attention is permutation-equivariant—without position info, `[A, B, C]` looks identical to `[C, B, A]` when querying from the end. GPT creates a learnable position table $\mathbf{P} \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}$ for positions `0, 1, ..., T-1` and adds it to the token embeddings:

     $$
     \mathbf{H}_0 = \mathbf{E}[\mathbf{X}] + \mathbf{P}[0:T]
     $$

     *(During backprop,* $\nabla_{\mathbf{P}} \mathcal{L}$ *sums the upstream gradient across the batch dimension* $B$*).*

  2. **Multi-Head Attention (MHA):** Multiple heads ($H$ heads, each of dimension $d_k = d_{\text{model}} / H$) allow the model to attend to multiple relationships simultaneously (e.g., syntax, local neighbors, long-range references):

     * Project $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ to shape $(B, T, d_{\text{model}})$.

     * Reshape to $(B, T, H, d_k)$ and transpose axes `(0, 2, 1, 3)` $\to$ $(B, H, T, d_k)$.

     * Run batched scaled dot-product causal attention (NumPy broadcasts `@` across both $B$ and $H$ automatically!).

     * Transpose back `(0, 2, 1, 3)` $\to$ $(B, T, H, d_k)$, reshape to $(B, T, d_{\text{model}})$, and multiply by output projection matrix $\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

* **Project Specification:**

  * **What the script should do:** Build a self-contained model with `PositionalEmbedding` and `MultiHeadAttention` and train it to **Reverse a Sequence** (a task impossible without positional awareness).

  * **Training Data:** 500 sequences of length 7: `[a, b, c, SEP, c, b, a]` where `a, b, c` $\in \{0, 1, 2, 3\}$ and `SEP = 4`. Compute loss only on the last 3 positions.

  * **Test Values:** Prompt `[1, 3, 0, 4]` $\to$ autoregressively generate 3 tokens. Expected output: `[0, 3, 1]`.

## Phase 6: Assembling & Training the Full GPT

### Project 17: GELU Activation & The Transformer Decoder Block

* **New Concept:** The Position-Wise Feed-Forward Network (MLP inside every block), the smooth **GELU** activation function, and assembling a complete Pre-LN Transformer Decoder Block.

* **Deep-Dive Theory:**
  Attention gathers information *across* tokens, while the **Feed-Forward Network (FFN)** processes that gathered information *within* each token independently. It expands the hidden size by $4\times$ ($d_{\text{model}} \to 4d_{\text{model}}$) and projects back down ($4d_{\text{model}} \to d_{\text{model}}$):

  $$
  \text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
  $$

  GPT uses **GELU (Gaussian Error Linear Unit)** instead of ReLU because it is smooth everywhere and preserves small negative gradients:

  $$
  \text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)
  $$

  *(Let* $u = \sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)$ *and* $t = \tanh(u)$*. By the chain rule,* $\text{GELU}'(x) = 0.5(1 + t) + 0.5x(1 - t^2)\sqrt{\frac{2}{\pi}}(1 + 3 \cdot 0.044715 x^2)$*).*

  **Complete Pre-LN Transformer Decoder Block:**

  $$
  \mathbf{x}_1 = \mathbf{x} + \text{MHA}(\text{LayerNorm}_1(\mathbf{x}))
  $$

  $$
  \mathbf{x}_2 = \mathbf{x}_1 + \text{FFN}(\text{LayerNorm}_2(\mathbf{x}_1))
  $$

* **Project Specification:**

  * **What the script should do:** Write a standalone script implementing a complete Pre-LN Transformer Decoder Block (`LayerNorm` + `MHA` + Residual + `LayerNorm` + `GELU FFN` + Residual) and verify its entire end-to-end analytical backward pass against finite differences.

  * **Training Data:** Random input $\mathbf{X} \in \mathbb{R}^{2 \times 5 \times 16}$ ($H=4$ heads) passed through one Transformer block with MSE loss.

  * **Test Values:** Confirm `Relative Error < 1e-7` with respect to input $\mathbf{X}$ and internal parameters.

### Project 18: Full GPT Architecture, Weight Tying & LR Scheduling

* **New Concept:** Stacking $L$ Transformer blocks into a full GPT architecture, **Weight Tying** between input embeddings and output logits, and Learning Rate Warmup + Cosine Decay.

* **Deep-Dive Theory:**

  1. **Full Forward Architecture:**

     * Token + Positional Embeddings: $\mathbf{H}_0 = \mathbf{E}[\mathbf{X}] + \mathbf{P}[0:T]$

     * Pass through $L$ stacked Transformer blocks: $\mathbf{H}_L = \text{Block}_L(\dots \text{Block}_1(\mathbf{H}_0))$

     * Final LayerNorm: $\mathbf{H}_{\text{norm}} = \text{LayerNorm}_f(\mathbf{H}_L)$

     * **Weight Tying:** Compute vocabulary logits by multiplying by the transpose of the token embedding table $\mathbf{E}^T$:

       $$
       \mathbf{Z} = \mathbf{H}_{\text{norm}} \mathbf{E}^T \in \mathbb{R}^{B \times T \times V}
       $$

       *(Why? Tokens with similar meanings should have similar rows in* $\mathbf{E}$ *both when entering the network and when predicting output probabilities! During backprop, remember to add the gradient* $\mathbf{G}_Z^T \mathbf{H}_{\text{norm}}$ *into* $\nabla_{\mathbf{E}}\mathcal{L}$ *alongside the embedding lookup gradient).*

  2. **Learning Rate Schedule:** Transformers destabilize if Adam takes large steps on step 1 when gradients are noisy. Linearly warm up $\eta$ from $0$ to $\eta_{\max}$ over the first 5% of steps, then decay smoothly with a cosine curve:

     $$
     \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\right)\right)
     $$

* **Project Specification:**

  * **What the script should do:** Write a standalone script implementing the complete multi-layer GPT forward/backward pass and cosine warmup schedule, and run a full-model finite-difference gradient check (including the tied embedding weights $\mathbf{E}$) on a tiny configuration ($L=2, H=2, d_{\text{model}}=8, V=6$).

  * **Training Data:** Random integer token batch $\mathbf{X} \in \{0..5\}^{2 \times 4}$ and shifted target $\mathbf{Y} \in \{0..5\}^{2 \times 4}$ with Cross-Entropy loss.

  * **Test Values:** Confirm `Relative Error < 1e-7` for $\mathbf{E}$, $\mathbf{P}$, and all block parameters, and verify that the scheduler rises linearly during warmup and decays smoothly to $\eta_{\min}$.

### Project 19: Training GPT on Real Text & Autoregressive Sampling

* **New Concept:** End-to-end language model training, **Temperature** scaling, **Top-**$k$ filtering, and autoregressive text generation.

* **Deep-Dive Theory:**
  Once your GPT model minimizes next-token cross-entropy loss on a text corpus, you generate new text one token at a time:

  1. Forward pass the current context tokens (cropping to the last $T_{\max}$ tokens if the sequence grows longer than the context window).

  2. Extract the logits $\mathbf{z} \in \mathbb{R}^V$ at the **last sequence position** (`T - 1`).

  3. **Temperature Scaling (**$\tau > 0$**):** Divide logits by $\tau$: $\mathbf{z}' = \mathbf{z} / \tau$.

     * $\tau < 1.0$ sharpens the probability distribution (more confident, coherent, conservative).

     * $\tau > 1.0$ flattens the distribution (more creative, surprising, chaotic).

  4. **Top-**$k$ **Filtering:** Keep only the $k$ highest logits and set all other logits to $-\infty$ so the model never accidentally picks low-probability gibberish from the long tail.

  5. Compute probabilities $\mathbf{p} = \text{Softmax}(\mathbf{z}')$, sample the next token ID via `np.random.choice(V, p=p)`, append it to the prompt, and repeat!

* **Project Specification:**

  * **What the script should do:** Build your complete, self-contained **Pure NumPy GPT** script (including BPE tokenization, $L = 2$ Transformer blocks, $H = 4$ heads, $d_{\text{model}} = 64$, context length $T = 32$, `vocab_size = 256`), train it end-to-end on real text, and generate new text with Temperature and Top-$k$ sampling.

  * **Training Data:** A public-domain text file of 15,000–40,000 characters (such as Shakespeare's *Romeo and Juliet* Act 1, or *Aesop's Fables*). Encode with BPE and slice into overlapping pairs $(\mathbf{X}, \mathbf{Y})$ of length $T = 32$ where $\mathbf{Y}$ is shifted 1 token to the right of $\mathbf{X}$.

  * **Test Values:**

    1. **Loss Curve:** Watch training cross-entropy loss drop from initial random guessing ($\ln(256) \approx 5.54$) down below $1.3$.

    2. **Text Generation:** Feed a 3-word seed prompt from your dataset with `temperature = 0.7` and `top_k = 20`, and generate 120 new tokens. Verify that your pure-NumPy model produces readable words, line breaks, and punctuation matching the style of your training text.