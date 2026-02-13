Contributing
============

Thank you for your interest in contributing to TESLearn!

Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/teslearn.git
      cd teslearn

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

Code Style
----------

We use Black for code formatting:

.. code-block:: bash

   black teslearn/

We use mypy for type checking:

.. code-block:: bash

   mypy teslearn/

Testing
-------

Run tests with pytest:

.. code-block:: bash

   pytest

With coverage:

.. code-block:: bash

   pytest --cov=teslearn

Documentation
-------------

Build documentation:

.. code-block:: bash

   cd docs
   make html

View built docs:

.. code-block:: bash

   open _build/html/index.html

Pull Request Process
--------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Run tests and linting
7. Submit a pull request

Coding Guidelines
-----------------

- Follow PEP 8 style guide
- Use type hints
- Write docstrings in NumPy format
- Keep functions focused and small
- Add examples to docstrings

Reporting Issues
----------------

Please include:

- Python version
- TESLearn version
- Operating system
- Steps to reproduce
- Expected vs actual behavior

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.