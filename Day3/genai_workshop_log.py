import logging

# Dependency: For demonstration, let's log successful import of 'requests'
try:
    import requests
    dependency_status = "requests module imported successfully"
except ImportError as e:
    dependency_status = f"Failed to import requests: {str(e)}"

# Set up logging to file (not console)
logging.basicConfig(
    filename='genai_workshop.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

logging.info("This is a log message from Gen AI Workshop!")
logging.info(f"Dependency Check: {dependency_status}")
