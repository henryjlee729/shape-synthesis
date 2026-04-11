import os
import pygame
import pygame_gui

try:
    import viewer.viewer as viewer
except ModuleNotFoundError:
    import viewer

DEFAULT_FILE = './test_chair_data/1a6f615e8b1b5ae4dbbc9440457e303e.pts'

def run_menu():
    pygame.init()

    width, height = 1000, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Neural 3D Shape Generator')

    manager = pygame_gui.UIManager((width, height))
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont('Arial', 42, bold=True)
    info_font = pygame.font.SysFont('Arial', 22)

    generate_rect = pygame.Rect(width // 2 - 100, height // 2 + 60, 200, 60)
    open_rect = pygame.Rect(width // 2 - 100, height // 2 + 140, 200, 60)

    generate_button = pygame_gui.elements.UIButton(
        relative_rect=generate_rect,
        text='Generate',
        manager=manager,
    )

    open_button = pygame_gui.elements.UIButton(
        relative_rect=open_rect,
        text='Open File',
        manager=manager,
    )

    selected_file = DEFAULT_FILE
    file_dialog = None

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

            manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == generate_button:
                    return selected_file

                if event.ui_element == open_button and file_dialog is None:
                    file_dialog = pygame_gui.windows.UIFileDialog(
                        rect=pygame.Rect(150, 100, 700, 500),
                        manager=manager,
                        window_title='Select a 3D file',
                        initial_file_path=os.getcwd(),
                        allow_existing_files_only=True,
                    )

            if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                selected_file = event.text
                file_dialog = None

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == file_dialog:
                    file_dialog = None

        manager.update(time_delta)
        screen.fill((20, 20, 30))
        title = title_font.render('Neural 3D Shape Generator', True, (255, 255, 255))
        title_rect = title.get_rect(center=(width // 2, height // 2 - 140))
        screen.blit(title, title_rect)

        subtitle = info_font.render('Select a .pts, .ply, or .obj file', True, (190, 190, 205))
        subtitle_rect = subtitle.get_rect(center=(width // 2, height // 2 - 90))
        screen.blit(subtitle, subtitle_rect)

        label = info_font.render('Selected file:', True, (220, 220, 230))
        label_rect = label.get_rect(center=(width // 2, height // 2 - 20))
        screen.blit(label, label_rect)

        shown_path = selected_file if len(selected_file) < 80 else '...' + selected_file[-77:]
        file_text = info_font.render(shown_path, True, (150, 200, 255))
        file_rect = file_text.get_rect(center=(width // 2, height // 2 + 10))
        screen.blit(file_text, file_rect)

        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()
    return None

if __name__ == '__main__':
    file_path = run_menu()

    if file_path is not None:
        viewer.main(file_path)
