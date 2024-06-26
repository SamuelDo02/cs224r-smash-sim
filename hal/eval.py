import melee


def run_episode() -> None:
    console = melee.Console(path="/SlippiOnline/")

    controller = melee.Controller(console=console, port=1)
    controller_human = melee.Controller(console=console, port=2, type=melee.ControllerType.GCN_ADAPTER)

    console.run()
    console.connect()

    controller.connect()
    controller_human.connect()

    while True:
        gamestate = console.step()
        # Press buttons on your controller based on the GameState here!


if __name__ == "__main__":
    main()
