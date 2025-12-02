program quadratic_map
    implicit none

    call generate_figure1_data()
    call generate_figure2_data()
    call generate_figure3_data()

    print *, "All data files generated successfully!"

contains

    ! The quadratic map g(x) = 4x(1-x)
    function g(x) result(y)
        real(8), intent(in) :: x
        real(8) :: y
        y = 4.0d0 * x * (1.0d0 - x)
    end function g

    ! Generate data for Figure 1: The quadratic map
    subroutine generate_figure1_data()
        integer, parameter :: n = 1000
        real(8) :: x, y
        integer :: i, unit_num

        print *, "Generating data for Figure 1..."

        open(newunit=unit_num, file='data_figure1_fortran.csv', status='replace')
        write(unit_num, '(A)') 'x,g_x,line_45'

        do i = 0, n-1
            x = real(i, 8) / real(n-1, 8)
            y = g(x)
            write(unit_num, '(F16.10,A,F16.10,A,F16.10)') x, ',', y, ',', x
        end do

        close(unit_num)
        print *, "Figure 1 data written to data_figure1_fortran.csv"
    end subroutine generate_figure1_data

    ! Generate data for Figure 2: Time series when x0 = 0.3
    subroutine generate_figure2_data()
        integer, parameter :: n = 150
        real(8), parameter :: x0 = 0.3d0
        real(8) :: x
        integer :: t, unit_num

        print *, "Generating data for Figure 2..."

        open(newunit=unit_num, file='data_figure2_fortran.csv', status='replace')
        write(unit_num, '(A)') 't,x_t'

        x = x0
        do t = 0, n-1
            write(unit_num, '(I0,A,F16.10)') t, ',', x
            x = g(x)
        end do

        close(unit_num)
        print *, "Figure 2 data written to data_figure2_fortran.csv"
    end subroutine generate_figure2_data

    ! Generate data for Figure 3: Histogram with 100,000 observations
    subroutine generate_figure3_data()
        integer, parameter :: n = 100001
        real(8), parameter :: x0 = 0.3d0
        real(8) :: x
        integer :: t, unit_num

        print *, "Generating data for Figure 3..."

        open(newunit=unit_num, file='data_figure3_fortran.csv', status='replace')
        write(unit_num, '(A)') 'x_t'

        x = x0
        do t = 0, n-1
            write(unit_num, '(F16.10)') x
            x = g(x)
        end do

        close(unit_num)
        print *, "Figure 3 data written to data_figure3_fortran.csv"
    end subroutine generate_figure3_data

end program quadratic_map
