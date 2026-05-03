FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HF Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /app

# Install Python deps as root (writes to system site-packages for all users)
COPY spaces/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Pre-download argostranslate EN→ES language pack so first-run is fast.
# Runs as root with HOME=/home/user so packages land in the user's data dir,
# then chown so the non-root runtime user can read and update the index.
RUN python - <<'EOF'
import argostranslate.package, argostranslate.translate
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
pkg = next((p for p in available if p.from_code == "en" and p.to_code == "es"), None)
if pkg:
    path = pkg.download()
    argostranslate.package.install_from_path(path)
    print("argostranslate EN→ES installed")
EOF
RUN chown -R user:user /home/user/.local

# Copy project source
COPY --chown=user:user . /app

USER user

EXPOSE 7860
CMD ["python", "spaces/app.py"]
