import melee
import melee.enums as enums
from loguru import logger

class MatchupMenuHelper:
    """Helper class for navigating menus automatically."""
    
    def __init__(self, controller_1, controller_2, character_1, character_2, stage, opponent_cpu_level=9):
        self.controller_1 = controller_1
        self.controller_2 = controller_2
        self.character_1 = character_1
        self.character_2 = character_2
        self.stage = stage
        self.opponent_cpu_level = opponent_cpu_level
        self._player_1_character_selected = False

    def select_character_and_stage(self, gamestate):
        """Navigate through menus automatically."""
        if not hasattr(self, 'menu_helper'):
            self.menu_helper = melee.menuhelper.MenuHelper()
            
        if gamestate.menu_state == enums.Menu.MAIN_MENU:
            logger.info("At main menu, choosing versus mode...")
            self.menu_helper.choose_versus_mode(gamestate=gamestate, controller=self.controller_1)
            
        elif gamestate.menu_state == enums.Menu.CHARACTER_SELECT:
            logger.info("At character select, choosing characters...")
            # Choose character for player 1 (human controlled bot)
            self.menu_helper.choose_character(
                character=self.character_1,
                gamestate=gamestate,
                controller=self.controller_1,
                cpu_level=0,  # human player
                costume=0,
                swag=False,
                start=False,
            )
            
            # Choose character for player 2 (CPU)
            if self.character_2 is not None:
                self.menu_helper.choose_character(
                    character=self.character_2,
                    gamestate=gamestate,
                    controller=self.controller_2,
                    cpu_level=self.opponent_cpu_level,
                    costume=1,
                    swag=False,
                    start=True,
                )
                
        elif gamestate.menu_state == enums.Menu.STAGE_SELECT:
            logger.info("At stage select, choosing stage...")
            if self.stage is not None:
                self.menu_helper.choose_stage(
                    stage=self.stage, 
                    gamestate=gamestate, 
                    controller=self.controller_1, 
                    character=self.character_1
                )
        
        elif gamestate.menu_state == enums.Menu.POSTGAME_SCORES:
            self.menu_helper.skip_postgame(controller=self.controller_1)