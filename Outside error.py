def calculate_outside_error_for_aes(test_blocks, predicted_blocks):
    """
    :param test_blocks: list of blocks from the test set, where each block is a list of binary numbers 1/0 with
    length 128 :param predicted_blocks: list of blocks which are predicted from the model, where each block is with
    length 128 and each i-th block corresponds to the i-th block from the test set :return: double number,
    in the range 0-1, that represents the outside error from the prediction of AES
    """
    if len(test_blocks) != len(predicted_blocks):
        raise Exception
    sum = 0
    for i in range(0, len(test_blocks)):
        if len(test_blocks[i]) != len(predicted_blocks[i]):
            raise Exception
        for j in range(0, len(test_blocks[i])):
            sum += ((test_blocks[i][j] + predicted_blocks[i][j]) % 2)
    return sum / (len(test_blocks) * len(test_blocks[0]))


def calculate_inside_error_for_aes(test_blocks, predicted_blocks):
    """
    :param test_blocks: list of blocks from the test set, where each block is a list of binary numbers 1/0 with
    length 128 :param predicted_blocks: list of blocks which are predicted from the model, where each block is with
    length 128 and each i-th block corresponds to the i-th block from the test set :return: double number,
    in the range 0-1, that represents the inside error from the prediction of AES
    """
    if len(test_blocks) != len(predicted_blocks):
        raise Exception
    count = 0
    for i in range(0, len(test_blocks)):
        if match(test_blocks[i], predicted_blocks[i]):
            count += 1
    return count / len(test_blocks)


def match(block1, block2):
    """
    :param block1: left block that should be checked if its a complete match with the right block (list of binary numbers)
    :param block2: right block (list of binary numbers)
    :return: boolean that symbolizes that the block1 and block2 are a complete match
    """
    if len(block1) != len(block2):
        raise Exception
    for i in range(0, len(block1)):
        if block1[i] != block2[i]:
            return False
    return True