# Model artifacts

Trained model files are not tracked by Git by default.

After training, the pipeline writes:

- `best_model_wrapper.joblib`
- `best_model_metadata.json`

The wrapper can be loaded with:

```python
from radiomics_lymphedema.wrapper import load_best_model
wrapper = load_best_model("models/best_model_wrapper.joblib")
```
