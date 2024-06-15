def draw_line():
    """
    Returns a line break for user interface.
    """
    return "-" * 80


def financial_warning():
    """
    Displays warning string to user.
    """
    warning = (
        f"{draw_line()}\n"
        " *WARNING* This programme is for educational purposes only.\n"
        " This is NOT financial advice and should not be used for trading.\n"
        f"{draw_line()}"
    )
    return warning
