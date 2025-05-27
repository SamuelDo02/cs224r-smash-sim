import melee
import melee.enums as enums
from loguru import logger

class Controller:
    """A controller that performs basic actions."""
    
    def __init__(self, controller):
        self.controller = controller
        
    def act(self, action):
        """Act on the controller."""
        # Reset controller inputs
        self.controller.release_all()
        
        try:
            self.controller.tilt_analog(enums.Button.BUTTON_MAIN, 1, 0.5)                
        except Exception as e:
            # If any error occurs, just do nothing this frame
            logger.warning(f"Controller error: {e}")
            pass