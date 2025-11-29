program quadratic_map
    implicit none

    ! Quadratic map: g(x) = 4x(1-x)

    ! Main program
    call generate_function_data()
    call generate_trajectory(0.3d0, 150, 'trajectory_fortran.csv')
    call generate_histogram_data(0.3d0, 100000, 'histogram_data_fortran.csv')

    print *, 'All Fortran data files generated successfully!'

contains

    ! Quadratic map function: g(x) = 4x(1-x)
    subroutine generate_function_data()
        integer, parameter :: num_points = 1000
        real(8) :: x, g_x
        integer :: i, unit_num

        open(newunit=unit_num, file='quadratic_function_fortran.csv', status='replace')
        write(unit_num, '(A)') 'x,g_x'

        do i = 0, num_points
            x = real(i, 8) / real(num_points, 8)
            g_x = quadratic_map_func(x)
            write(unit_num, '(F20.15,A,F20.15)') x, ',', g_x
        end do

        close(unit_num)
        print *, 'Generated quadratic_function_fortran.csv'
    end subroutine generate_function_data

    ! Generate trajectory data for time series
    subroutine generate_trajectory(x0, n, filename)
        real(8), intent(in) :: x0
        integer, intent(in) :: n
        character(len=*), intent(in) :: filename
        real(8) :: x
        integer :: t, unit_num

        open(newunit=unit_num, file=filename, status='replace')
        write(unit_num, '(A)') 't,x'

        x = x0
        do t = 0, n
            write(unit_num, '(I0,A,F20.15)') t, ',', x
            x = quadratic_map_func(x)
        end do

        close(unit_num)
        print *, 'Generated ', trim(filename)
    end subroutine generate_trajectory

    ! Generate trajectory data for histogram (long time series)
    subroutine generate_histogram_data(x0, n, filename)
        real(8), intent(in) :: x0
        integer, intent(in) :: n
        character(len=*), intent(in) :: filename
        real(8) :: x
        integer :: t, unit_num

        open(newunit=unit_num, file=filename, status='replace')
        write(unit_num, '(A)') 'x'

        x = x0
        do t = 0, n
            write(unit_num, '(F20.15)') x
            x = quadratic_map_func(x)
        end do

        close(unit_num)
        print *, 'Generated ', trim(filename)
    end subroutine generate_histogram_data

end program quadratic_map
