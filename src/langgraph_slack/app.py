# /src/langgraph_slack/app.py Register shutdown hooks
def register_app_shutdown_hooks(app):
    """
    Register shutdown hooks for the application.
    
    Args:
        app: FastAPI or Flask application
    """
    from pro.agents.sql.bq_routing import _stop_cache_cleanup, cleanup_sql_engines
    from pro.runtime.singletons import reset_singletons
    
    # For FastAPI
    if hasattr(app, 'on_event'):
        @app.on_event("shutdown")
        def shutdown_hook():
            _stop_cache_cleanup()
            reset_singletons()
    
    # For Flask
    elif hasattr(app, 'teardown_appcontext'):
        @app.teardown_appcontext
        def shutdown_hook(exception=None):
            _stop_cache_cleanup()
            reset_singletons()
    
    # Register with atexit as fallback
    import atexit
    atexit.register(_stop_cache_cleanup)
    atexit.register(reset_singletons)

# Register shutdown hooks with the app
register_app_shutdown_hooks(app)