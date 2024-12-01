import sys

def error_message_detail(err,err_detail:sys):
    _,_,exc_tb = err_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    err_message = 'Error occcured in python script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(err)
    )
    return err_message

class CustomException(Exception):
    def __init__(self,err_message,err_detail:sys):
        super().__init__(err_message)
        self.err_message= error_message_detail(err_message, err_detail=err_detail)
    
    def __str__(self):
        return self.err_message