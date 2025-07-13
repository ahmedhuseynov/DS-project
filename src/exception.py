import sys
import logging

def error_mesage(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    error_message = f"Error occurred in script: [{exc_tb.tb_frame.f_code.co_filename}] at line number: [{exc_tb.tb_lineno}] with error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_mesage(error_message, error_detail)

    def __str__(self):
        return self.error_message

