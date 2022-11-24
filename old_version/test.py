from traceback import print_tb
import chardet
from hashlib import sha3_256
## For change hex 2 base58 str ##
import base58
import binascii
#################################

#Size standard : Byte
blocksize = 1048576
data = 20
randomsize = 108
num_block = 0x64

### For random number ##
import os
########################

block_chain = []
trans_info = []


def genesis():
    """create genesis block"""
    data = 'Genesis'
    TxID = sha3_256(data.encode('utf-8')).hexdigest()
    TxID = TxID[-20:]
    prev_hash = ''
    current_hash = create_hash(data, prev_hash)
    #여기에 trans_data -> 넘겨주기(txid, data)
    add_block(TxID, data, prev_hash, current_hash)#trans -> txid, data 처리 예정이므로 지우기

def create_hash(data: str, prev_hash: bytes):
    """create hash"""
    new_hash = sha3_256((data+prev_hash).encode('utf-8')).hexdigest()
    return new_hash

def add_block(TxID, data, prev_hash, current_hash):
    """add block in block_chain"""
    block_chain.append((TxID, data, prev_hash, current_hash))

def add_pre_status_block(TxID:str, data: str):
    """add pre status block, add real block chain"""
    _, _, _, prev_hash = block_chain[-1]
    current_hash = create_hash(data, prev_hash)
    add_block(TxID, data, prev_hash, current_hash)
    
def show_block_chain():#3-2-2번, 160비트로 지정해야함.
    """show block chain model"""
    for i, (TxID, data, prev_hash, current_hash) in enumerate(block_chain):
        i = "0x"+(hex(i))[2:].zfill(20)
        print(f'\nBlock : {i}\n'
              f'trans_info : {trans_info}\n'
              f'TxID : {base58.b58encode(TxID)}\n'
              f'data : {base58.b58encode(data)}\n'
              f'Previous hash : {base58.b58encode(prev_hash)}\n'
              f'Current hash : {base58.b58encode(current_hash)}')
        
def valid_block_chain(): # 3-5번 검증 완료
    """validation of block_chain"""
    for i in range(1, len(block_chain)):
        TxID, data, prev_hash, current_hash = block_chain[i]
        TxID, last_data, last_prev_hash, last_current_hash = block_chain[i - 1]
        if prev_hash != last_current_hash:
            print(f"block {i} prev_hash not equal block {i-1} current_hash.\n"
                  f"Plz check {prev_hash.hex()} not equal {last_current_hash.hex()}")
            return False
        if (last_current_hash) != (temp := create_hash(last_data, last_prev_hash)):
            valid_fail(i - 1, last_current_hash, temp[0], temp[1])
            return False
        if (current_hash) != (temp := create_hash(data, prev_hash)):
            valid_fail(i, current_hash, temp[0], temp[1])
            return False
    print('validation result :')
    return True

def valid_fail(block_num, ori_hash, new_hash):
    print(f"block {block_num} validation falied. \n"
          f"check this {ori_hash} != \n {new_hash}")
    

genesis()

for i in range(num_block):#data 부분임 block 별로 0x1,2,3,...99까지 올라가야함.
    for j in range(2):
        random_size = 864
        random_number_bytes = os.urandom(random_size // 8)#3-1-3
        #print('1 : ',random_number_bytes)
        random_number_bytes = binascii.hexlify(random_number_bytes)
        random_number_bytes_108 = random_number_bytes.decode('utf-8')
        #print(random_number_bytes)
        #print(type(random_number_bytes_108))
        random_number_bytes = sha3_256(random_number_bytes).hexdigest()
        random_number_bytes = random_number_bytes[-20:]
        TxID = sha3_256(random_number_bytes.encode('utf-8')).hexdigest()
        TxID = TxID[-20:]
        #print('this is TxID',TxID)
    trans_info.append(base58.b58encode(TxID+random_number_bytes_108))
    add_pre_status_block(TxID, random_number_bytes)#조건 3-3-3

show_block_chain()
print()
print(valid_block_chain())

'''
    trans_qua='+'
    trans_info.append(TxID + trans_qua + random_number_bytes)
'''