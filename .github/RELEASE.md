## Releasing a New Version

1. **Update version number** in both:
   - `pyproject.toml` (`[project] version = ...`)
   - `src/grounded_memory/__init__.py` (`__version__ = ...`)

2. **Update `CHANGELOG.md`** (if you maintain one)

3. **Commit and push:**
   ```bash
   git add pyproject.toml src/grounded_memory/__init__.py
   git commit -m "release: bump version to X.Y.Z"
   git push
   ```

4. **Create a Git tag:**
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. **Create a GitHub Release** from the tag:
   - Go to GitHub → Releases → Draft a new release
   - Choose the tag you just pushed
   - Add release notes
   - Click **Publish release**

6. **CI/CD handles the rest:**
   - The `publish.yml` workflow triggers automatically on release publication
   - It builds the wheel + sdist and uploads to PyPI

7. **Verify on PyPI:**
   ```bash
   pip install grounded-memory==X.Y.Z
   ```
