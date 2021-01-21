# paths
dataset_folder                     = "./dataset"
foils_dat_path                     = "./Foils DB/dat"
foils_pkl_path                     = "./Foils DB/pkl"
foils_bmp_path                     = "./Foils DB/bmp"
weights_path                       = "./weights"

# foil preprocessing params
n_points_Re                        = 16
n_points_alfa                      = 32
n_foil_params                      = 8
n_foil_points                      = 128
alfa_min                           = -2.1
alfa_max                           = 8.
re_min                             = 40000
re_max                             = 200000
max_nans_in_curve                  = 0.6
flap_position                      = 0.7
xfoil_max_iterations               = 64
foil_exception_list                = ['goe388.pkl' , 'goe451.pkl' , 'eiffel428.pkl' , 'eiffel430.pkl', 'ebambino7.pkl', 'ea81006.pkl',
                                      'goe802a.pkl', 'goe802b.pkl', 'naca63a210.pkl', 'naca63206.pkl', 'rc1064c.pkl'  , 'saratov.pkl',
                                      'ua79sff.pkl']

# bitmaps generation params
bitmap_outputs                     = [(256, 1024), (512, 512)]
zoom_coef                          = 2048./28.45
n_points_interpolate_for_bmp       = 10000

# training params
train_percentage                   = 0.75
val_percentage                     = 0.9

# prediction params
n_points_in_predicted_dat          = 256
weights_file                       = 'train v4 bs12 2020-12-31 11-47.h5'
yellow_threshold                   = 0.25