load('../data/mat/test')

test_indx = 100;

test_image = reshape( test_x(test_indx,:,:), [28,28] );
test_label = test_y( test_indx );


imshow( test_image )
fprintf( 'Label is: %i\n', test_label );