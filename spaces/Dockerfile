FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl build-essential libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HF Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /app

# Install all deps in one pass — torch/torchvision/torchaudio are pinned to
# CUDA 12.1 wheels in requirements.txt; --extra-index-url makes pip find them.
COPY spaces/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Pre-download argostranslate EN→ES language pack so first-run is fast
RUN python - <<'EOF'
import argostranslate.package
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
pkg = next((p for p in available if p.from_code == "en" and p.to_code == "es"), None)
if pkg:
    path = pkg.download()
    argostranslate.package.install_from_path(path)
    print("argostranslate EN-ES installed")
EOF
RUN chown -R user:user /home/user/.local

# Copy project source
COPY --chown=user:user . /app

USER user

EXPOSE 7860
CMD ["python", "spaces/app.py"]
