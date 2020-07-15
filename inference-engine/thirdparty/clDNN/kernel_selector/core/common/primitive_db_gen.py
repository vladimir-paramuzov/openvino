#!/usr/bin/python

# To add new kernel please add a .cl file to kernels directory
# the database name will be the part of the file name up to first '.' character
# the trailing characters are a tag to allow multiple primitive implementations

from __future__ import print_function
import os
import argparse
import glob
import ntpath
import re

import gzip
import textwrap

class OpenCL2CHeaders(object):

    def __init__(self, kernels_folder, out_path, out_file_name):
        self.kernels_folder = os.path.abspath(kernels_folder)
        self.out_path = os.path.abspath(out_path)
        self.out_file_name = out_file_name
        self.include_files = {}

    def convert(self):
        res = '// This file is autogenerated by primitive_db_gen.py, all changes to this file will be undone\n\n'
        filelist = glob.glob(os.path.join(self.kernels_folder, "*.cl"))

        out_file_name = os.path.join(self.out_path, self.out_file_name)
        out_file = open(out_file_name, 'w')
        out_file.write('{\n')
        for filename in filelist:
            try:
                print('processing {}'.format(filename))
                name = ntpath.basename(filename)
                self.include_files[filename] = {}
                #kernel_name = name[:name.find('.')]
                kernel_name = name[:name.find('.cl')]
                res = self.cl_file_to_str1(filename)
                compressed_value = gzip.compress(bytes(res,'utf-8'))
                # print(type(compressed_value))
                out_file.write('{{"{}",std::vector<uint8_t>{{\n'.format(kernel_name))
                s = ''
                l = 0
                for x in compressed_value:
                    ss = '0x{:02x},'.format(x)
                    l = l + len(ss)
                    s = s + ss
                    if l >= 160:
                        s = s + '\n'
                        l = 0

                out_file.write(s)
                out_file.write('}}}},\n'.format(kernel_name))
            except:
               pass
        out_file.write('\n}')

        res = ''
        for filename in filelist:
            #try:
                print('processing {}'.format(filename))
                res += self.cl_file_to_str(filename)
            #except:
            #    pass

        out_file_name = os.path.join(self.out_path, self.out_file_name+"1")
        print(out_file_name)
        #with open(out_file_name, 'r') as out_file:
        #    old_content = out_file.read()
        #if old_content != res:
        #print('Replacing old DB')
        with open(out_file_name, 'w') as out_file:
            out_file.write(res)
        #with open(out_file_name, 'r') as out_file:
        #    old_content = out_file.read()
        #if old_content != res:
        #print('Replacing old DB')
        # with open(out_file_name, 'w') as out_file:
            # out_file.write(res)

    def append_file_content(self, filename, origin_file):
        res = ""
        content = []
        with open(filename) as f:
            content += f.readlines()
        for line in content:
            if line.startswith('#include'):
                include_file_name = line.strip().split('"')[1].strip()
                full_path_include = os.path.abspath(os.path.join(os.path.dirname(filename), include_file_name))
                if full_path_include not in self.include_files[origin_file]:
                    self.include_files[origin_file][full_path_include] = True
                    res += self.append_file_content(full_path_include, origin_file)
                    res += "\n"
                continue
            res += '{}\n'.format(line.rstrip())
        return res

    def cl_file_to_str1(self, filename):
        name = ntpath.basename(filename)
        self.include_files[filename] = {}
        #kernel_name = name[:name.find('.')]
        kernel_name = name[:name.find('.cl')]
        res = '\n'
        content = self.append_file_content(filename, filename)
        max_lines = 200
        max_characters = 16350
        characters = 1  # Newline character above

        comment_regexp = re.compile(r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?', re.DOTALL | re.MULTILINE)

        def comment_replacer(match):
            begin, mid, end = match.group(1,2,3)
            if mid is None:
                return ''
            elif begin is not None or end is not None:
                return ''
            elif '\n' in mid:
                return '\n'
            else:
                return ' '

            return

        # Remove comments
        content = comment_regexp.sub(comment_replacer, content)
        # Remove empty lines
        content = os.linesep.join([s for s in content.splitlines() if s])
        # Remove multiple spaces
        content = re.sub(' +', ' ', content)
        for i, line in enumerate(content.split('\n')):
            if (i + 1) % max_lines == 0 or characters + len(line) + 1 > max_characters:
                characters = 0
            res += line + '\n'
            characters += len(line) + 1

        return res

    def cl_file_to_str(self, filename):
        name = ntpath.basename(filename)
        self.include_files[filename] = {}
        #kernel_name = name[:name.find('.')]
        kernel_name = name[:name.find('.cl')]
        res = '{{"{}",\n(std::string) R"__krnl(\n'.format(kernel_name)
        content = self.append_file_content(filename, filename)
        max_lines = 200
        max_characters = 16350
        characters = 1  # Newline character above

        comment_regexp = re.compile(r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?', re.DOTALL | re.MULTILINE)

        def comment_replacer(match):
            begin, mid, end = match.group(1,2,3)
            if mid is None:
                return ''
            elif begin is not None or end is not None:
                return ''
            elif '\n' in mid:
                return '\n'
            else:
                return ' '

            return

        # Remove comments
        content = comment_regexp.sub(comment_replacer, content)
        # Remove empty lines
        content = os.linesep.join([s for s in content.splitlines() if s])
        # Remove multiple spaces
        content = re.sub(' +', ' ', content)
        for i, line in enumerate(content.split('\n')):
            if (i + 1) % max_lines == 0 or characters + len(line) + 1 > max_characters:
                res += ')__krnl"\n + R"__krnl('
                characters = 0
            res += line + '\n'
            characters += len(line) + 1

        res += ')__krnl"}},\n\n'.format(kernel_name)

        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-kernels', required=True, metavar='PATH', help='The absolute path to OpenCL kernels folder')
    ap.add_argument('-out_path', required=True, metavar='PATH', help='The absolute path to dump file')
    ap.add_argument('-out_file_name', required=True, metavar='PATH', help='dump file name')
    args = ap.parse_args()

    converter = OpenCL2CHeaders(args.kernels, args.out_path, args.out_file_name)
    converter.convert()

if __name__ == '__main__':
    main()