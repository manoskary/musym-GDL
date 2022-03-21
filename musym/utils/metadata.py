
MOZART_PIANO_SONATAS = [
	'K279-1', 'K279-2', 'K279-3', 'K280-1', 'K280-2',
	'K280-3', 'K281-1', 'K281-2', 'K281-3', 'K282-1',
	'K282-2', 'K282-3', 'K283-1', 'K283-2', 'K283-3',
	'K284-1', 'K284-2', 'K284-3', 'K309-1', 'K309-2',
	'K309-3', 'K310-1', 'K310-2', 'K310-3', 'K311-1',
	'K311-2', 'K311-3', 'K330-1', 'K330-2', 'K330-3',
	'K331-1', 'K331-3', 'K332-1', 'K332-2',
	'K332-3', 'K333-1', 'K333-2', 'K333-3', 'K457-1',
	'K457-2', 'K457-3', 'K533-1', 'K533-2', 'K533-3',
	'K545-1', 'K545-2', 'K545-3', 'K570-1', 'K570-2',
	'K570-3', 'K576-1', 'K576-2', 'K576-3'
	]
# Problematic mps
# 'K331-2',


MOZART_STRING_QUARTETS = [
	'k590-01', 'k155-02', 'k156-01', 'k080-02', 'k172-01',
	'k171-01', 'k172-04', 'k157-01', 'k589-01', 'k458-01',
	'k169-01', 'k387-01', 'k158-01', 'k157-02', 'k171-03',
	'k159-02', 'k428-02', 'k173-01', 'k499-03', 'k156-02',
	'k168-01', 'k080-01', 'k421-01', 'k171-04', 'k168-02',
	'k428-01', 'k499-01', 'k172-02', 'k465-04', 'k155-01',
	'k465-01', 'k159-01'
	]


BACH_FUGUES = [
	'wtc1f01', 'wtc1f07', 'wtc1f15', 'wtc1f13',
	'wtc1f06', 'wtc1f03', 'wtc1f02', 'wtc1f18',
	'wtc1f17', 'wtc1f09', 'wtc1f24', 'wtc1f10',
	'wtc1f22', 'wtc1f16', 'wtc1f12', 'wtc1f23',
	'wtc1f19', 'wtc1f05', 'wtc1f14', 'wtc1f04',
	'wtc1f08', 'wtc1f20', 'wtc1f21',
	]

HAYDN_STRING_QUARTETS = [
	'haydn_op064_no06_mv01_1770', 'haydn_op050_no06_mv01_1756',
	'haydn_op020_no06_mv02_1740', 'haydn_op020_no01_mv04_1733',
	'haydn_op076_no05_mv02_1776', 'haydn_op020_no05_mv01_1739',
	'haydn_op017_no02_mv01_1728', 'haydn_op033_no02_mv01_1743',
	'haydn_op054_no03_mv01_1761', 'haydn_op050_no01_mv01_1748',
	'haydn_op017_no06_mv01_1732', 'haydn_op064_no04_mv01_1768',
	'haydn_op064_no04_mv04_1769', 'haydn_op017_no05_mv01_1731',
	'haydn_op064_no03_mv04_1767', 'haydn_op054_no02_mv01_1760',
	'haydn_op055_no02_mv02_1764', 'haydn_op064_no03_mv01_1766',
	'haydn_op033_no03_mv03_1744', 'haydn_op074_no01_mv01_1772',
	'haydn_op054_no01_mv01_1758', 'haydn_op076_no02_mv01_1774',
	'haydn_op033_no05_mv02_1747', 'haydn_op055_no01_mv02_1763',
	'haydn_op054_no01_mv02_1759', 'haydn_op050_no02_mv01_1750',
	'haydn_op050_no03_mv04_1752', 'haydn_op020_no04_mv04_1738',
	'haydn_op033_no01_mv03_1742', 'haydn_op033_no05_mv01_1746',
	'haydn_op050_no06_mv02_1757', 'haydn_op020_no03_mv04_1736',
	'haydn_op076_no04_mv01_1775', 'haydn_op050_no05_mv04_1755',
	'haydn_op033_no01_mv01_1741', 'haydn_op054_no03_mv04_1762',
	'haydn_op050_no04_mv01_1753', 'haydn_op050_no02_mv04_1751',
	'haydn_op017_no01_mv01_1727', 'haydn_op033_no04_mv01_1745',
	'haydn_op017_no03_mv04_1729', 'haydn_op050_no01_mv04_1749',
	'haydn_op055_no03_mv01_1765', 'haydn_op074_no01_mv02_1773',
	'haydn_op020_no03_mv03_1735'
	]


PIANO = BACH_FUGUES + MOZART_PIANO_SONATAS

QUARTETS = MOZART_STRING_QUARTETS + HAYDN_STRING_QUARTETS

MOZART = MOZART_STRING_QUARTETS + MOZART_PIANO_SONATAS

MIX = PIANO + QUARTETS

FILE_LIST = [
	'note-during-note.csv', 'note-follows-note.csv',
	'note-follows-rest.csv', 'note-onset-note.csv',
	'note.csv', 'rest-follows-note.csv', 'rest.csv'
	]

BASIS_FN = [
	'onset_feature.score_position', 'duration_feature.duration', 'fermata_feature.fermata',
	'grace_feature.n_grace', 'grace_feature.grace_pos', 'onset_feature.onset',
	'polynomial_pitch_feature.pitch', 'grace_feature.grace_note',
	'relative_score_position_feature.score_position', 'slur_feature.slur_incr',
	'slur_feature.slur_decr', 'time_signature_feature.time_signature_num_1',
	'time_signature_feature.time_signature_num_2', 'time_signature_feature.time_signature_num_3',
	'time_signature_feature.time_signature_num_4', 'time_signature_feature.time_signature_num_5',
	'time_signature_feature.time_signature_num_6', 'time_signature_feature.time_signature_num_7',
	'time_signature_feature.time_signature_num_8', 'time_signature_feature.time_signature_num_9',
	'time_signature_feature.time_signature_num_10', 'time_signature_feature.time_signature_num_11',
	'time_signature_feature.time_signature_num_12', 'time_signature_feature.time_signature_num_other',
	'time_signature_feature.time_signature_den_1', 'time_signature_feature.time_signature_den_2',
	'time_signature_feature.time_signature_den_4', 'time_signature_feature.time_signature_den_8',
	'time_signature_feature.time_signature_den_16', 'time_signature_feature.time_signature_den_other',
	'vertical_neighbor_feature.n_total', 'vertical_neighbor_feature.n_above', 'vertical_neighbor_feature.n_below',
	'vertical_neighbor_feature.highest_pitch', 'vertical_neighbor_feature.lowest_pitch',
	'vertical_neighbor_feature.pitch_range'
	]