import numpy as np
import re
import random
import math
from collections import Counter


embedding_dim = 50

# Read dataset
with open("data.txt", "r") as file:
    dataset = file.read()

lst = re.findall(r"\b\w+\b", dataset)

word_freq = Counter(lst)
vocab = list(word_freq.keys())  # Convert to list for consistent indexing
vocab_len = len(vocab)

token_idx = {token: idx for idx, token in enumerate(vocab)}

# Assigning random values to the embedding matrices
int_matrix_embedding = np.random.randn(vocab_len, embedding_dim)
weight_matrix_embedding = np.random.randn(vocab_len, embedding_dim)

# Skip Gram model
def skip_gram_ns(lst, window_size=5, negative_num=5):
    all_token = list(token_idx.keys())
    training_pair = []
    
    for center_idx in range(len(lst)):
        center_token = lst[center_idx]
        
        # Skip if center token not in vocabulary
        if center_token not in token_idx:
            continue
        
        actual_context = []
        for offset in range(-window_size, window_size + 1):
            context_idx = center_idx + offset
            if offset == 0 or context_idx < 0 or context_idx >= len(lst):
                continue
            if lst[context_idx] in token_idx:  # Check if context token is in vocab
                actual_context.append(lst[context_idx])
        
        for context_token in actual_context:
            # Positive example - use center token as target, context as context
            target_vector = int_matrix_embedding[token_idx[center_token]]
            context_vector = weight_matrix_embedding[token_idx[context_token]]  
            training_pair.append((target_vector, context_vector, 1))
            
            # Negative sampling
            for _ in range(negative_num):
                while True:
                    random_token = random.choice(all_token)
                    if random_token != center_token and random_token not in actual_context:
                        break
                
                target_vector = int_matrix_embedding[token_idx[center_token]]
                negative_vector = weight_matrix_embedding[token_idx[random_token]]  
                training_pair.append((target_vector, negative_vector, 0))
    
    return training_pair

def word2vec(lst, learning_rate=0.01, epochs=10):
    epoch_losses = []  
    
    for epoch in range(epochs):
        training_examples = skip_gram_ns(lst)
        epoch_loss = 0.0  
        
        for target_vec, context_vec, label in training_examples:
            dot_product = np.dot(target_vec, context_vec)  
            predicted_prob = 1 / (1 + math.exp(-dot_product))
            
            # Numerical stability
            epsilon = 1e-15
            predicted_prob = max(epsilon, min(1 - epsilon, predicted_prob))
            
            # Calculate loss (binary cross-entropy)
            loss = -(label * math.log(predicted_prob) + (1 - label) * math.log(1 - predicted_prob))
            epoch_loss += loss
            
            # Calculate gradient
            error = predicted_prob - label
            
            # Update vectors (create copies to avoid modifying during calculation)
            target_gradient = error * context_vec
            context_gradient = error * target_vec
            
            # Apply gradients
            target_vec -= learning_rate * target_gradient
            context_vec -= learning_rate * context_gradient
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / len(training_examples)
        epoch_losses.append(avg_epoch_loss)
        
        # Print progress every few epochs
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.6f}")
    
    return int_matrix_embedding, weight_matrix_embedding, epoch_losses

# Train the model
trained_target_embedding, trained_context_embedding, losses = word2vec(lst, learning_rate=0.01, epochs=40)



print(f"\nTraining completed!")
print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Total loss reduction: {losses[0] - losses[-1]:.6f}")
print(f"Loss reduction percentage: {((losses[0] - losses[-1]) / losses[0]) * 100:.2f}%")

# Test with words
#some Relevent-Irrelevent word to evaluate model.
#Relevent words: Aurelia-rooftops, Talin-Chronicon, shard - descendants, clock‑maker - gears
#Irrelevent words: rooftops-veins, 	Talin-decay, Chronicon-cat
word = "Talin"                                     
word1 = "decay"

print(f"{word}: {trained_target_embedding[token_idx[word]]}")
print(f"{word1}: {trained_target_embedding[token_idx[word1]]}")
