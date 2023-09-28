# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##


class CSVReader:
    
    def __init__(self, file_name, encoding):
        self.file = open(file_name, 'rb')
        self.encoding = encoding
        self.lines = []
        self.file.seek(0, 2)
        self.file_size = self.file.tell()
        self.file.seek(0, 0)
        self.read_header()
        self.read_file()
        
    def read_header(self):
        line = self.file.readline().decode(self.encoding)
        self.header = line.strip().split(",")
        self.header = [ s.strip() for s in self.header ]
    
    def read_file(self):
        start = self.file.tell()
        while start < self.file_size:
            line = self.file.readline()
            end = self.file.tell()
            self.lines.append( (start, end - start) )
            start = end
    
    def __del__(self):
        del(self.lines)
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        start, length = self.lines[index]
        self.file.seek(start, 0)
        line = self.file.read(length)
        line = line.decode(self.encoding).strip().split(",")
        line = [ s.strip() for s in line ]
        
        obj = {}
        for i in range(len(line)):
            key = self.header[i]
            value = line[i]
            obj[key] = value
        
        return obj
