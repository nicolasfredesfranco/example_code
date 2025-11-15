# E-Commerce Search Engine Optimization

## Harvard Business School - Retrieval Assignment

This repository contains my solution to the HBS e-commerce search engine optimization challenge. The original baseline implementation and my enhanced version with detailed analyses and improvements are both included.

---

## Assignment Overview

This assignment focuses on improving an e-commerce search engine using the WANDS (Wayfair ANnotation Dataset) dataset. The task involves enhancing a baseline TF-IDF retrieval system to achieve better Mean Average Precision at 10 (MAP@10) scores.

### Original Challenge Requirements

The assignment provided a baseline Python notebook implementing a simple e-commerce search engine using TF-IDF vectorization. The baseline system achieved a MAP@10 score of 0.29, which is considered low compared to production e-commerce systems (typically 0.6-0.8).

**Key Tasks:**

1. **Propose and Implement Improvements**: Analyze the baseline system and suggest modifications to increase the MAP@10 score. The focus is on demonstrating sound reasoning and methodology rather than achieving a specific score threshold.

2. **Alternative Evaluation Metrics**: The baseline treats partial matches as irrelevant, which penalizes the model too strictly. Implement a fairer evaluation function that leverages partial match information and justify the chosen approach with tradeoffs.

3. **Implementation Choice** (Option A or B):
   - **Option A**: Implement at least one improvement from Task 1 and demonstrate measurable performance gains
   - **Option B**: Refactor the code to be more object-oriented, flexible, and production-ready

**Technical Specifications:**

- Dataset: WANDS (Wayfair ANnotation Dataset)
- Evaluation Metric: Mean Average Precision at 10 (MAP@10)
- Required packages: pandas 1.5.3, scikit-learn 1.3.2, numpy 1.3.0
- Allowed resources: Open-source packages and publicly accessible APIs
- Performance target: MAP@10 greater than 0.3 (no upper limit expectation)

---

## Repository Contents

### Files

- **`original_HBS_retrieval_assignment.ipynb`**: The original baseline implementation provided by HBS
  - Simple TF-IDF vectorization approach
  - Basic MAP@10 evaluation
  - Serves as the starting point for improvements

- **`HBS_retrieval_assignment.ipynb`**: My enhanced solution with comprehensive improvements
  - Detailed information-theoretic analysis of TF-IDF
  - Advanced preprocessing and feature engineering
  - Deep learning-based query-product mapping
  - Alternative evaluation metrics
  - Extensive documentation and visualizations

- **`WANDS/`**: The Wayfair ANnotation Dataset
  - `dataset/product.csv`: Product catalog with descriptions and metadata
  - `dataset/query.csv`: User search queries
  - `dataset/label.csv`: Human-annotated relevance labels (Exact, Partial, Irrelevant)

- **`query_evaluation_results_nn.csv`**: Evaluation results from the neural network model

- **`query_product_mapping_model_final.pth`**: Trained PyTorch model weights

---

## My Solution: Key Improvements

### 1. Information-Theoretic Foundation

Developed a comprehensive theoretical framework for understanding TF-IDF from an information theory perspective:
- Term frequency as local probability distribution
- IDF as global surprisal (self-information)
- Vector space as information-weighted representation
- Provides mathematical foundation for subsequent improvements

### 2. Advanced Product Representation (528 dimensions)

**Feature Engineering:**
- Multiple Correspondence Analysis (MCA) for categorical features
- TF-IDF text embeddings combined with categorical encodings
- Normalized vector representations for cosine similarity
- Removed non-semantic features (ratings, review counts) to focus on content

**Rationale:** Combining categorical structure with semantic text creates richer product representations that capture both taxonomy and content.

### 3. Query Processing Pipeline

**Preprocessing Steps:**
- Spell checking using `pyspellchecker` to handle user typos
- Stop word removal to focus on meaningful terms
- Tokenization and text normalization
- Vocabulary alignment with product space

**Impact:** Cleaner queries lead to more accurate retrieval, especially for real-world queries with spelling errors.

### 4. Dimensionality Reduction via SVD

- Compressed TF-IDF vectors from 44,307 to 512 dimensions
- Retains 95%+ of semantic information
- Reduces computational complexity
- Removes noise and focuses on latent semantic patterns

**Analysis:** Includes visualization of explained variance and information retention to justify the chosen dimensionality.

### 5. Deep Learning Neural Network

**Architecture:**
- Input: 512-dimensional query embeddings (from SVD)
- Hidden layer: 4,096 neurons with LeakyReLU activation
- Output: 528-dimensional product space with unit normalization
- Loss function: Triplet loss + MSE for multi-level relevance
- Optimizer: Adam with learning rate 0.001 and weight decay

**Training Strategy:**
- Phase 1: Cross-validation to determine optimal epochs (prevents overfitting)
- Phase 2: Full dataset training with optimal hyperparameters
- GPU acceleration for efficient training
- Xavier initialization for better convergence

**Result:** Learned non-linear mapping between query and product spaces that captures semantic relationships.

### 6. Weighted MAP@10 Evaluation

**Modified Metric:**
- Exact matches: Score = 1.0
- Partial matches: Score = 0.5
- Irrelevant: Score = 0.0

**Justification:**
- Provides more nuanced evaluation
- Recognizes value in partial matches
- Better reflects real-world relevance gradients
- Gives clearer signal for model improvements

**Tradeoffs:**
- Pro: Less strict penalization, more realistic
- Pro: Encourages models to get close even if not perfect
- Con: May inflate perceived performance
- Con: Less interpretable than binary evaluation

### 7. Comprehensive System Demonstration

Includes end-to-end demonstrations with:
- Example queries (including intentional typos)
- Step-by-step processing visualization
- Similarity score analysis
- Product recommendation results
- Comparison between different approaches

---

## Technical Implementation

### Dependencies

```bash
pip install pandas==1.5.3
pip install scikit-learn==1.3.2
pip install numpy==1.3.0
pip install matplotlib
pip install prince
pip install torch torchvision
pip install nltk
pip install pyspellchecker
```

### Reproducibility

The solution is fully reproducible:
1. All random seeds are set for consistency
2. Complete dataset is included in the repository
3. Trained model weights are saved and can be loaded
4. All cells can be run sequentially from a fresh kernel

### GPU Acceleration

The neural network training automatically detects and uses GPU if available:
- Tested on NVIDIA GeForce GTX 1660 Ti
- Fallback to CPU if GPU is unavailable
- Training time: ~5-10 minutes on GPU, ~30-40 minutes on CPU

---

## Results Summary

### Performance Metrics

**Baseline (Original TF-IDF):**
- MAP@10: 0.29

**Neural Network Model:**
- MAP@10: 0.0435 (strict evaluation with exact matches only)
- Weighted MAP@10: 0.0059 (with partial matches weighted at 0.5)

**Note:** The neural network model showed lower MAP@10 than the baseline on this specific evaluation. This is discussed in detail in the notebook with analysis of:
- Why the score decreased despite theoretical improvements
- Potential overfitting and training data quality issues
- Trade-offs between semantic understanding and exact matching
- Recommendations for further improvements

### Key Insights

1. **Vocabulary Mismatch**: User queries often contain out-of-vocabulary terms that the product catalog does not include

2. **Training Data Quality**: The model is only as good as the labeled training data; more diverse training examples could improve performance

3. **Semantic vs. Lexical Matching**: Deep learning models capture semantic similarity but may miss lexical exact matches that TF-IDF naturally handles

4. **Hybrid Approach Potential**: Combining TF-IDF and neural embeddings could leverage strengths of both approaches

---

## Analysis and Documentation

The enhanced notebook includes:

- **Theoretical Foundation**: Information-theoretic analysis of TF-IDF and vector spaces
- **Data Analysis**: Comprehensive exploration of products, queries, and labels
- **Dimensionality Analysis**: PCA/SVD analysis with variance explained plots
- **Model Architecture**: Detailed explanation of neural network design choices
- **Training Visualization**: Loss curves, convergence analysis, overfitting detection
- **Error Analysis**: Examples of successful and failed retrievals
- **Comparison Studies**: Multiple evaluation metrics and model comparisons
- **Production Considerations**: Logging, error handling, batch processing

---

## Future Improvements

### Recommended Next Steps

1. **Hybrid Retrieval System**:
   - Combine TF-IDF (for lexical matching) with neural embeddings (for semantic matching)
   - Implement ensemble scoring with learned weights

2. **Advanced Embeddings**:
   - Use pre-trained language models (BERT, RoBERTa, SBERT)
   - Fine-tune on domain-specific e-commerce data
   - Leverage transfer learning for better semantic understanding

3. **Query Understanding**:
   - Query expansion using synonyms and related terms
   - Intent classification (navigational vs. informational)
   - Entity recognition for product attributes

4. **Ranking Optimization**:
   - Learning-to-rank algorithms (LambdaMART, RankNet)
   - Incorporate user behavior signals (clicks, purchases)
   - Personalization based on user history

5. **Data Augmentation**:
   - Generate synthetic training examples
   - Use data augmentation techniques (backtranslation, paraphrasing)
   - Active learning to identify and label hard examples

6. **Feature Engineering**:
   - Include product popularity metrics
   - Add temporal features (seasonality, trends)
   - Incorporate user context (location, device, session history)

7. **Production Optimization**:
   - Implement caching for common queries
   - Use approximate nearest neighbor search (FAISS, Annoy)
   - Deploy as microservice with proper monitoring
   - A/B testing framework for continuous improvement

---

## Assignment Response Summary

### Question 1: Proposed Updates to Increase MAP@10

Implemented comprehensive improvements including:
- Information-theoretic analysis of the baseline
- Feature engineering with categorical encodings
- Query preprocessing with spell checking
- Dimensionality reduction via SVD
- Deep learning neural network for query-product mapping

**Reasoning**: Each component addresses specific limitations of the baseline system, from vocabulary coverage to semantic understanding.

### Question 2: Alternative Evaluation Metrics

Implemented weighted MAP@10 that assigns:
- Exact matches: 1.0
- Partial matches: 0.5
- Irrelevant: 0.0

**Justification**: Provides more nuanced evaluation that recognizes partial relevance rather than treating it as completely irrelevant.

**Tradeoffs**: More realistic but potentially less interpretable than binary evaluation.

### Question 3: Implementation (Option A)

Chose Option A and implemented multiple improvements:
- Complete data analysis and preprocessing pipeline
- Neural network model with proper training methodology
- Comprehensive evaluation and comparison
- Full system demonstration with example queries

**Documentation**: Extensive markdown cells and code comments explain the thought process, design decisions, and results interpretation.

---

## Technical Notes

### Computational Requirements

- **Memory**: Minimum 8GB RAM (16GB recommended)
- **GPU**: Optional but recommended for neural network training
- **Storage**: ~500MB for dataset and model weights
- **Runtime**: ~15-30 minutes for complete notebook execution

### Code Quality

The solution includes:
- Modular functions with comprehensive docstrings
- Error handling and validation
- Progress indicators for long-running operations
- Visualization of intermediate results
- Saved artifacts for reproducibility

### Limitations and Considerations

1. **Overfitting Risk**: Deep learning model may overfit on limited training data
2. **Cold Start**: New products without embeddings require reprocessing
3. **Query Vocabulary**: Out-of-vocabulary queries need special handling
4. **Computational Cost**: Neural network inference is slower than TF-IDF
5. **Interpretability**: Deep learning results are less interpretable than TF-IDF

---

## Author

**Nicolas Fredes Franco**

This solution was developed as part of the Harvard Business School retrieval assignment.

## Contact and Attribution

The WANDS dataset is provided by Wayfair and is used under their license terms.

For questions or discussions about this implementation, please refer to the detailed documentation within the notebook itself.

---

## References

- Wayfair ANnotation Dataset (WANDS): https://github.com/wayfair/WANDS
- TF-IDF: Term Frequency-Inverse Document Frequency
- SVD: Singular Value Decomposition
- MCA: Multiple Correspondence Analysis
- MAP: Mean Average Precision

---

## License

This project follows the licensing terms of the WANDS dataset. Please refer to the WANDS repository for specific license information.
